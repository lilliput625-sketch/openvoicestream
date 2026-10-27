from __future__ import annotations

from unittest.mock import MagicMock

from server.core.codec_cpp_tts import CodecCppTTSBackend, CodecCppTTSConfig


def test_voicedesign_payload_uses_instructions_without_a_named_voice():
    backend = CodecCppTTSBackend(CodecCppTTSConfig(
        mode="voicedesign",
        model_id="qwen3-tts-voicedesign",
        request_model="qwen3-tts-voicedesign",
    ))

    payload = backend._payload(
        "Hello from VoiceDesign.",
        language="en",
        speaker="narrator",
        voice="ryan",
        instruct="A calm, warm adult voice, speaking slowly.",
        response_format="pcm",
    )

    assert payload["input"] == "Hello from VoiceDesign."
    assert payload["model"] == "qwen3-tts-voicedesign"
    assert "voice" not in payload
    assert payload["instructions"] == (
        "Speak in English. A calm, warm adult voice, speaking slowly."
    )
    assert backend.supports_hot_reload is True


def test_managed_voicedesign_process_is_started_and_stopped(monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(
        "server.core.codec_cpp_tts.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    backend = CodecCppTTSBackend(CodecCppTTSConfig(
        mode="voicedesign",
        model_id="qwen3-tts-voicedesign",
        request_model="qwen3-tts-voicedesign",
        manage_process=True,
        process_command=("tts-server", "--model", "voice.gguf"),
        startup_timeout_s=1,
    ))
    monkeypatch.setattr(backend, "is_ready", lambda: True)

    backend.preload()
    assert backend.is_ready() is True
    backend.unload()

    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_managed_auxiliary_process_is_reported_and_stopped(monkeypatch):
    model = MagicMock()
    model.poll.return_value = None
    adapter = MagicMock()
    adapter.poll.return_value = None
    processes = iter((model, adapter))
    monkeypatch.setattr(
        "server.core.codec_cpp_tts.subprocess.Popen",
        lambda *args, **kwargs: next(processes),
    )
    monkeypatch.setattr("server.core.codec_cpp_tts.socket.create_connection", lambda *args, **kwargs: MagicMock())
    backend = CodecCppTTSBackend(CodecCppTTSConfig(
        mode="voicedesign",
        manage_process=True,
        process_command=("tts-server",),
        auxiliary_command=("adapter",),
        auxiliary_ready_port=8624,
        startup_timeout_s=1,
    ))
    monkeypatch.setattr(backend, "is_ready", lambda: True)

    backend.preload()
    backend.unload()

    adapter.terminate.assert_called_once()
    model.terminate.assert_called_once()


def test_public_controls_apply_dsp_wrapper(monkeypatch):
    backend = CodecCppTTSBackend(CodecCppTTSConfig(
        mode="voicedesign",
        model_id="qwen3-tts-voicedesign",
        request_model="qwen3-tts-voicedesign",
    ))
    monkeypatch.setattr(backend, "_request", lambda payload, **kwargs: iter((b"abc", b"def")))

    observed = {}

    class FakeShifter:
        def __init__(self, *, sample_rate, speed, pitch_shift, channels):
            observed.update(
                sample_rate=sample_rate,
                speed=speed,
                pitch_shift=pitch_shift,
                channels=channels,
            )

        def push(self, chunk):
            return b"x" + chunk

        def flush(self):
            return b"tail"

    monkeypatch.setattr("voxedge.audio.rate.TTSRateShifter", FakeShifter)
    output = list(backend.generate_streaming(
        "hello",
        speed=1.5,
        pitch_shift=2.0,
    ))

    assert observed == {
        "sample_rate": 24000,
        "speed": 1.5,
        "pitch_shift": 2.0,
        "channels": 1,
    }
    assert output == [b"xabc", b"xdef", b"tail"]
