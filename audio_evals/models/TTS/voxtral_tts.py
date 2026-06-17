"""
Voxtral 4B TTS wrapper.

Voxtral is served via ``vllm-omni`` (a vllm fork with audio support). The
isolated launcher in ``audio_evals/lib/Voxtral/main.py`` brings up an HTTP
server inside its own venv and prints ``PORT:<base_url>`` once the server
is healthy. This wrapper then issues OpenAI-compatible
``POST /v1/audio/speech`` requests to synthesise audio.

Reference:
    https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
"""

import json
import logging
import os
import select
import tempfile
import threading
from typing import Dict, Optional

import requests
import soundfile as sf

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


def _extract_payload(prompt):
    if isinstance(prompt, dict):
        return dict(prompt)
    if isinstance(prompt, list):
        payload = {}
        for message in prompt:
            for item in message.get("contents", []):
                if item["type"] == "text":
                    payload[item.get("key", "text")] = item["value"]
                elif item["type"] == "audio":
                    payload[item.get("key", "prompt_audio")] = item["value"]
        return payload
    return {"text": str(prompt)}


def _response_suffix(response_format: str) -> str:
    return {
        "wav": ".wav",
        "mp3": ".mp3",
        "flac": ".flac",
        "opus": ".opus",
        "aac": ".aac",
        "pcm16": ".pcm",
    }.get(response_format, f".{response_format}" if response_format else ".wav")


@isolated("audio_evals/lib/Voxtral/main.py")
class VoxtralTTS(OfflineModel):
    """Voxtral 4B TTS client backed by an isolated vllm-omni server."""

    def __init__(
        self,
        path: str,
        startup_timeout: int = 1800,
        request_timeout: int = 1800,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_model_len: int = 0,
        gpu_memory_utilization: float = 0.0,
        extra_args: str = "",
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "startup_timeout": startup_timeout,
        }
        if max_model_len:
            self.command_args["max_model_len"] = max_model_len
        if gpu_memory_utilization:
            self.command_args["gpu_memory_utilization"] = gpu_memory_utilization
        if extra_args:
            self.command_args["extra_args"] = extra_args

        # ``model`` is required by the OpenAI-compatible /v1/audio/speech
        # endpoint and must match the model id served by vllm-omni
        # (vllm-omni reports the path/HF id used at launch).
        self._model_id = path
        self.request_timeout = request_timeout
        self.base_url = None
        self.speech_url = None
        self._stdout_thread = None
        self._stderr_thread = None
        super().__init__(is_chat=True, sample_params=sample_params)

    def _start_pipe_drain(self, stream_name: str, log_method):
        thread_attr = f"_{stream_name}_thread"
        if getattr(self, thread_attr) is not None:
            return

        def _drain():
            stream = getattr(self.process, stream_name)
            while True:
                try:
                    line = stream.readline()
                except Exception:
                    return
                if not line:
                    return
                log_method(
                    "Voxtral vllm launcher %s: %s",
                    stream_name,
                    line.rstrip(),
                )

        thread = threading.Thread(target=_drain, daemon=True)
        setattr(self, thread_attr, thread)
        thread.start()

    def _start_pipe_drains(self):
        self._start_pipe_drain("stdout", logger.debug)
        self._start_pipe_drain("stderr", logger.debug)

    def _drain_stderr_tail(self, max_bytes: int = 4000) -> str:
        """Best-effort: collect whatever the launcher already wrote on stderr.

        Used for diagnostics when the vllm-omni launcher dies before printing
        the ``PORT:<base_url>`` handshake line. Non-blocking via ``select``.
        """
        chunks = []
        try:
            while True:
                rlist, _, _ = select.select([self.process.stderr], [], [], 0)
                if not rlist:
                    break
                line = self.process.stderr.readline()
                if not line:
                    break
                chunks.append(line)
                if sum(len(c) for c in chunks) >= max_bytes:
                    break
        except Exception:
            pass
        tail = "".join(chunks)
        if len(tail) > max_bytes:
            tail = "...<truncated>...\n" + tail[-max_bytes:]
        return tail

    def _ensure_server_ready(self):
        if self.speech_url is not None:
            return

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1800
            )
            if not rlist:
                raise RuntimeError(
                    "Voxtral vllm-omni startup timeout after 1800 seconds"
                )

            if self.process.poll() is not None:
                stderr_tail = self._drain_stderr_tail()
                raise RuntimeError(
                    "Voxtral vllm-omni launcher exited early with code "
                    f"{self.process.returncode}.\n"
                    f"--- launcher stderr tail ---\n{stderr_tail}"
                )

            for stream in rlist:
                if stream == self.process.stdout:
                    line = self.process.stdout.readline().strip()
                    if not line:
                        continue
                    if line.startswith("PORT:"):
                        self.base_url = line[len("PORT:"):].strip().rstrip("/")
                        self.speech_url = f"{self.base_url}/v1/audio/speech"
                        self._start_pipe_drains()
                        logger.info(
                            "Voxtral vllm-omni is ready at %s",
                            self.speech_url,
                        )
                        return
                    logger.info(line)
                elif stream == self.process.stderr:
                    err = self.process.stderr.readline().strip()
                    if err:
                        logger.info("Voxtral launcher stderr: %s", err)

    def wait_until_ready(self):
        """Public hook used by ``IsolatedModelPool`` to pre-warm the instance.

        Blocks until the underlying vllm-omni HTTP server has printed
        ``PORT:<base_url>`` (i.e. is healthy). This guarantees that, once the
        pool finishes constructing all instances, dataset workers will never
        race against a half-booted (or crashed) launcher.
        """
        self._ensure_server_ready()

    def _build_request_payload(self, payload):
        text = payload.pop("text", "").strip()
        if not text:
            raise ValueError("Voxtral TTS requires a non-empty `text` field.")

        # Voxtral uses preset voices; drop reference-audio fields if a
        # generic voice-cloning prompt template injects them.
        payload.pop("prompt_audio", None)
        payload.pop("prompt_text", None)
        payload.pop("reference_audio", None)
        # ``language`` is informational only for downstream evaluators.
        payload.pop("language", None)

        voice = payload.pop("voice", "casual_male")
        response_format = str(payload.pop("response_format", "wav")).lower()

        request_payload = {
            "input": text,
            "model": self._model_id,
            "voice": voice,
            "response_format": response_format,
        }
        # Forward remaining fields so callers may pass extra knobs (speed,
        # etc.) without code changes.
        request_payload.update(payload)
        return request_payload, response_format

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        self._ensure_server_ready()

        payload = _extract_payload(prompt)
        payload.update(kwargs)
        request_payload, response_format = self._build_request_payload(payload)

        enable_rtf = int(os.environ.get("ENABLE_RTF", "0")) == 1
        start_time = None
        if enable_rtf:
            import time

            start_time = time.time()

        response = requests.post(
            self.speech_url,
            json=request_payload,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Voxtral vllm-omni request failed: "
                f"status={response.status_code}, body={response.text}"
            )

        with tempfile.NamedTemporaryFile(
            suffix=_response_suffix(response_format), delete=False
        ) as f:
            f.write(response.content)
            output_path = f.name

        if not enable_rtf:
            return output_path

        import time

        inference_time = time.time() - start_time
        audio_duration = 0.0
        try:
            info = sf.info(output_path)
            if info.samplerate > 0:
                audio_duration = info.frames / info.samplerate
        except Exception:
            logger.warning(
                "Failed to compute audio duration for %s", output_path
            )
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
        return json.dumps(
            {"audio": output_path, "RTF": rtf}, ensure_ascii=False
        )
