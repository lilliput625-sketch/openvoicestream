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
import wave
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Any, Callable, Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodecCppTTSConfig:
    server_url: str = "http://127.0.0.1:8623"
    model_id: str = "qwen3-tts-customvoice"
    request_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    default_voice: str = "ryan"
    sample_rate: int = 24000
    read_chunk_bytes: int = 4096
    read_timeout_s: float = 1.0


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
    """Stream the 1.7B CustomVoice model through a loopback codec.cpp server."""

    supports_hot_reload = False

    def __init__(self, config: CodecCppTTSConfig):
        self._config = config
        parsed = urlsplit(config.server_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("codec.cpp server_url must be an http:// URL")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._base_path = parsed.path.rstrip("/")
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_tts_1_7b_customvoice_codec_cpp"

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_SPEAKER,
            TTSCapability.MULTI_LANGUAGE,
        }

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def _connection(self, timeout: float = 120.0) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def is_ready(self) -> bool:
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
        if not self.is_ready():
            raise RuntimeError(
                f"codec.cpp TTS server is not ready at {self._config.server_url}; "
                "start it with qwen-tts-model customvoice"
            )

    def unload(self) -> None:
        # The model process is deliberately managed by qwen-tts-model so it can
        # be stopped before switching backends and releasing shared RAM.
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
                headers={"Content-Type": "application/json", "Accept": "audio/pcm"},
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
        style = (instruct or "").strip()
        instruction_parts = []
        if lang_name and lang_key not in ("auto",):
            instruction_parts.append(f"Speak in {lang_name}.")
        if style:
            instruction_parts.append(style)
        requested_voice = voice or speaker
        if requested_voice is None and lang_name:
            requested_voice = _LANGUAGE_VOICES.get(lang_name.lower())
        if requested_voice is None:
            requested_voice = self._config.default_voice
        requested_voice = _SPEAKER_IDS.get(str(requested_voice), str(requested_voice))
        payload: dict[str, Any] = {
            "model": self._config.request_model,
            "input": text,
            "response_format": response_format,
        }
        if requested_voice:
            payload["voice"] = str(requested_voice)
        if instruction_parts:
            payload["instructions"] = " ".join(instruction_parts)
        return payload

    def generate_streaming(
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

    def synthesize(
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
