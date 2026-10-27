"""OpenVoiceStream adapter for codec.cpp's local OpenAI-compatible TTS server.

The codec.cpp process owns the model and GPU memory. This adapter only forwards
per-buffer requests and yields the server's raw 24 kHz mono PCM chunks.
"""
from __future__ import annotations

import http.client
import io
import json
import logging
import socket
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Callable, Iterator, Optional, Sequence

from voxedge.backends.base import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodecCppTTSConfig:
    server_url: str = "http://127.0.0.1:8623"
    mode: str = "customvoice"
    model_id: str = "qwen3-tts-customvoice"
    request_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    default_voice: str = "ryan"
    default_instructions: str = ""
    sample_rate: int = 24000
    read_chunk_bytes: int = 4096
    read_timeout_s: float = 1.0
    manage_process: bool = False
    process_command: Sequence[str] = ()
    process_cwd: Optional[str] = None
    process_log_path: Optional[str] = None
    startup_timeout_s: float = 180.0
    auxiliary_command: Sequence[str] = ()
    auxiliary_cwd: Optional[str] = None
    auxiliary_log_path: Optional[str] = None
    auxiliary_ready_port: Optional[int] = None


_LANGUAGE_NAMES = {
    "en": "English", "english": "English", "en-us": "English",
    "zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese",
    "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kr": "Korean", "korean": "Korean",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "ru": "Russian", "russian": "Russian",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "es": "Spanish", "spanish": "Spanish",
    "it": "Italian", "italian": "Italian",
}

_LANGUAGE_VOICES = {
    "english": "ryan", "chinese": "vivian", "japanese": "ono_anna",
    "korean": "sohee",
}
_SPEAKER_IDS = {
    "3065": "vivian", "3061": "ryan", "2861": "aiden",
    "3066": "serena", "2878": "dylan", "2875": "eric",
    "3010": "uncle_fu", "2873": "ono_anna", "2864": "sohee",
}


class CodecCppTTSBackend(TTSBackend):
    """Stream a 1.7B model through a loopback codec.cpp server."""

    supports_hot_reload = True

    def __init__(self, config: CodecCppTTSConfig):
        self._config = config
        parsed = urlsplit(config.server_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("codec.cpp server_url must be an http:// URL")
        if config.mode not in ("customvoice", "voicedesign"):
            raise ValueError("codec.cpp mode must be customvoice or voicedesign")
        if config.manage_process and not config.process_command:
            raise ValueError("codec.cpp managed process requires process_command")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._base_path = parsed.path.rstrip("/")
        self._ready = False
        self._process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._aux_process: Optional[subprocess.Popen] = None
        self._aux_log_handle = None

    @property
    def name(self) -> str:
        return f"qwen3_tts_1_7b_{self._config.mode}_codec_cpp"

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        capabilities = {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_LANGUAGE,
        }
        if self._config.mode == "customvoice":
            capabilities.add(TTSCapability.MULTI_SPEAKER)
        return capabilities

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def _connection(self, timeout: float = 120.0) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def is_ready(self) -> bool:
        if self._process is not None and self._process.poll() is not None:
            return False
        conn = self._connection(timeout=0.5)
        try:
            conn.request("GET", f"{self._base_path}/health")
            resp = conn.getresponse()
            body = resp.read(512)
            self._ready = resp.status == 200
            return self._ready
        except (OSError, http.client.HTTPException):
            self._ready = False
            return False
        finally:
            conn.close()

    def preload(self) -> None:
        if self._config.manage_process:
            if self._process is not None and self._process.poll() is None:
                return
            log_path = self._config.process_log_path
            if log_path:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = open(log_path, "a", buffering=1)
            stdout = self._log_handle if self._log_handle is not None else subprocess.DEVNULL
            stderr = subprocess.STDOUT if self._log_handle is not None else subprocess.DEVNULL
            self._process = subprocess.Popen(
                list(self._config.process_command),
                cwd=self._config.process_cwd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )

        deadline = time.monotonic() + self._config.startup_timeout_s
        while time.monotonic() < deadline:
            if self.is_ready():
                self._start_auxiliary()
                self._ready = True
                return
            if self._process is not None and self._process.poll() is not None:
                break
            time.sleep(0.5)
        rc = self._process.poll() if self._process is not None else None
        self.unload()
        raise RuntimeError(
            f"codec.cpp TTS server is not ready at {self._config.server_url} "
            f"(process_rc={rc}, log={self._config.process_log_path})"
        )

    def _start_auxiliary(self) -> None:
        command = list(self._config.auxiliary_command)
        if not command:
            return
        log_path = self._config.auxiliary_log_path
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._aux_log_handle = open(log_path, "a", buffering=1)
        stdout = self._aux_log_handle if self._aux_log_handle is not None else subprocess.DEVNULL
        stderr = subprocess.STDOUT if self._aux_log_handle is not None else subprocess.DEVNULL
        self._aux_process = subprocess.Popen(
            command,
            cwd=self._config.auxiliary_cwd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        port = self._config.auxiliary_ready_port
        if port is None:
            return
        deadline = time.monotonic() + self._config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._aux_process.poll() is not None:
                break
            try:
                with socket.create_connection((self._host, port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)
        rc = self._aux_process.poll()
        self._stop_auxiliary()
        raise RuntimeError(
            f"codec.cpp auxiliary process is not ready on {self._host}:{port} "
            f"(process_rc={rc}, log={self._config.auxiliary_log_path})"
        )

    def _stop_auxiliary(self) -> None:
        process = self._aux_process
        self._aux_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    logger.exception("codec.cpp auxiliary process cleanup failed")
        if self._aux_log_handle is not None:
            try:
                self._aux_log_handle.close()
            finally:
                self._aux_log_handle = None

    def unload(self) -> None:
        self._stop_auxiliary()
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    logger.exception("codec.cpp managed process cleanup failed")
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
        self._ready = False

    def _request(
        self, payload: dict[str, Any], *, stream: bool,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Iterator[bytes] | bytes:
        conn = self._connection()
        path = f"{self._base_path}/v1/audio/speech"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            conn.request(
                "POST", path, body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "audio/pcm",
                    "Connection": "close",
                },
            )
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(f"codec.cpp returned HTTP {resp.status}: {detail}")
            if not stream:
                try:
                    return resp.read()
                finally:
                    conn.close()
            if conn.sock is not None:
                conn.sock.settimeout(self._config.read_timeout_s)

            def chunks() -> Iterator[bytes]:
                try:
                    while True:
                        try:
                            # read1 returns one available HTTP chunk promptly,
                            # avoiding a full-size read wait between audio frames.
                            chunk = resp.read1(self._config.read_chunk_bytes)
                        except (socket.timeout, TimeoutError):
                            if should_cancel is not None and should_cancel():
                                return
                            continue
                        if not chunk:
                            break
                        yield chunk
                finally:
                    conn.close()

            return chunks()
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _cancelled(*signals: Any) -> bool:
        for signal in signals:
            checker = getattr(signal, "is_set", None)
            if callable(checker) and checker():
                return True
        return False

    def _payload(
        self, text: str, *, language: Optional[str], speaker: Optional[str],
        voice: Optional[str], instruct: Optional[str], response_format: str,
    ) -> dict[str, Any]:
        lang_key = (language or "").strip().lower()
        lang_name = _LANGUAGE_NAMES.get(lang_key)
        style = (instruct or self._config.default_instructions).strip()
        instruction_parts = []
        if lang_name and lang_key not in ("auto",):
            instruction_parts.append(f"Speak in {lang_name}.")
        if style:
            instruction_parts.append(style)
        payload: dict[str, Any] = {
            "model": self._config.request_model,
            "input": text,
            "response_format": response_format,
        }
        if self._config.mode == "customvoice":
            requested_voice = voice or speaker
            if requested_voice is None and lang_name:
                requested_voice = _LANGUAGE_VOICES.get(lang_name.lower())
            if requested_voice is None:
                requested_voice = self._config.default_voice
            requested_voice = _SPEAKER_IDS.get(str(requested_voice), str(requested_voice))
            if requested_voice:
                payload["voice"] = str(requested_voice)
        if instruction_parts:
            payload["instructions"] = " ".join(instruction_parts)
        return payload

    def _generate_streaming_impl(
        self, text: str, *, language: Optional[str] = None,
        speaker: Optional[str] = None, cancel_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[bytes]:
        cancel_event = kwargs.get("cancel_event")
        payload = self._payload(
            text,
            language=language,
            speaker=speaker,
            voice=kwargs.get("voice"),
            instruct=kwargs.get("instruct") or kwargs.get("tts_instruct"),
            response_format="pcm",
        )
        should_cancel = lambda: self._cancelled(cancel_token, cancel_event)
        chunks = self._request(payload, stream=True, should_cancel=should_cancel)
        assert not isinstance(chunks, bytes)
        for chunk in chunks:
            if self._cancelled(cancel_token, cancel_event):
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()
                return
            if chunk:
                yield chunk

    def _synthesize_impl(
        self, text: str, speaker_id: Optional[int] = None,
        speed: Optional[float] = None, pitch_shift: Optional[float] = None,
        language: Optional[str] = None, **kwargs: Any,
    ) -> tuple[bytes, dict]:
        speaker = kwargs.get("speaker") or kwargs.get("voice") or speaker_id
        payload = self._payload(
            text,
            language=language,
            speaker=str(speaker) if speaker is not None else None,
            voice=kwargs.get("voice"),
            instruct=kwargs.get("instruct") or kwargs.get("tts_instruct"),
            response_format="wav",
        )
        result = self._request(payload, stream=False)
        assert isinstance(result, bytes)
        duration = 0.0
        try:
            with wave.open(io.BytesIO(result), "rb") as wav:
                duration = wav.getnframes() / float(wav.getframerate())
        except (wave.Error, EOFError, ZeroDivisionError):
            pass
        return result, {
            "sample_rate": self.sample_rate,
            "duration_s": duration,
            "model_id": self.model_id,
        }
