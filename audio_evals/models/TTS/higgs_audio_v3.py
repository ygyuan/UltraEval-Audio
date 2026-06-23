"""
Higgs Audio v3 TTS wrapper.

Higgs Audio v3 (``bosonai/higgs-audio-v3-tts-4b``) is the latest standalone
TTS release from Boson AI. We serve it via ``vllm-omni`` (installed editable
from the local ``third_party/vllm-omni`` clone, which is the only place
where the v3 model class ``HiggsMultimodalQwen3ForConditionalGeneration``
is registered) and talk to it through the OpenAI-compatible audio speech
API ``POST /v1/audio/speech``.

The isolated launcher in ``audio_evals/lib/HiggsAudioV3/main.py`` boots
``vllm-omni serve <model> --omni --trust-remote-code`` inside its own venv
and prints ``PORT:<base_url>`` once the server is healthy. This wrapper
then issues ``POST /v1/audio/speech`` requests with the standard fields
(``input``, ``model``, ``response_format``) plus the v3-specific
``ref_audio`` (data URL, base64) / ``ref_text`` voice-clone fields.

References:
    https://huggingface.co/bosonai/higgs-audio-v3-tts-4b
    third_party/vllm-omni/recipes/BosonAI/Higgs-Audio-V3-TTS.md
    third_party/vllm-omni/examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py
"""

import base64
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


_AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


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


def _audio_data_url(path: str) -> str:
    suffix = os.path.splitext(path)[1].lower().lstrip(".")
    mime = _AUDIO_MIME.get(suffix, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _response_suffix(response_format: str) -> str:
    return {
        "wav": ".wav",
        "mp3": ".mp3",
        "flac": ".flac",
        "opus": ".opus",
        "aac": ".aac",
        "pcm": ".pcm",
        "pcm16": ".pcm",
    }.get(response_format, f".{response_format}" if response_format else ".wav")


# ``pre_command`` runs INSIDE the launcher's activated env, just before the
# regular ``pip install -r requirements.txt`` step.
#
# Higgs Audio v3 is served through the upstream ``vllm-omni`` project. The
# ``HiggsMultimodalQwen3ForConditionalGeneration`` model class was added in
# the vllm-omni commit ``82e74ce9`` (June 2026, "[TTS][New Model] support
# bosonai/higgs-audio-v3-tts-4b"); that code path is only compatible with
# ``vllm==0.22.x`` (it imports vllm 0.22-only symbols such as
# ``vllm.inputs.engine.TokensInput`` and ``vllm.v1.request.StreamingUpdate``
# in ``vllm_omni/patch.py`` and ``vllm_omni/inputs/data.py``). Older
# vllm-omni releases (v0.17 ~ v0.20) predate the Higgs V3 model and DO NOT
# register it. Older vllm releases (0.11.x / 0.18.x) lack the new internal
# API and cannot satisfy vllm-omni 0.22's imports.
#
# Hardware/driver requirement
# ---------------------------
# The published ``vllm==0.22.0`` wheel pins ``torch==2.11.0``, and the only
# torch 2.11 build on PyPI is a CUDA 13 build linked against
# ``libcudart.so.13``. Loading the wheel's ``vllm/_C.abi3.so`` therefore
# requires an NVIDIA driver that supports CUDA 13 (driver >= 575). On a host
# with driver 535 (CUDA 12.x) the wheel cannot initialise and the launcher
# will exit with ``OSError: libcudart.so.13: cannot open shared object file``
# regardless of how ``LD_LIBRARY_PATH`` is tweaked, because the runtime needs
# a CUDA 13 capable driver to back ``libcudart.so.13``.
#
# We therefore keep the version pin at the upstream recommended ``0.22.0`` so
# that on a properly provisioned host (driver >= 575, CUDA 13) this path
# works out of the box. Hosts stuck on driver 535 cannot run Higgs Audio v3
# and should either upgrade the driver or use a different TTS model.
#
# ``--no-deps`` is used to keep the launcher idempotent across runs: the
# cloned conda env (see ``registry/model/higgs_audio_v3.yaml``) already ships
# the rest of the recipe's runtime helpers (msgspec, prometheus-client,
# blake3, etc.); without ``--no-deps`` pip would also try to (re)install the
# pinned ``torch==2.11.0`` even when the env already has a working copy.
_HIGGS_V3_VLLM = "vllm==0.22.0"
_HIGGS_V3_PRE_COMMAND = f"pip install --no-deps {_HIGGS_V3_VLLM}"


@isolated(
    "audio_evals/lib/HiggsAudioV3/main.py",
    pre_command=_HIGGS_V3_PRE_COMMAND,
)
class HiggsAudioV3TTS(OfflineModel):
    """Higgs Audio v3 TTS client backed by an isolated vllm-omni server."""

    def __init__(
        self,
        path: str,
        startup_timeout: int = 1800,
        request_timeout: int = 1800,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_model_len: int = 0,
        gpu_memory_utilization: float = 0.0,
        max_new_tokens: int = 2048,
        seed: int = 42,
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

        # ``model`` field of /v1/audio/speech must match the model id served
        # by vllm-omni (it reports the path/HF id used at launch).
        self._model_id = path
        self.request_timeout = request_timeout
        self.default_max_new_tokens = max_new_tokens
        self.default_seed = seed

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
                    "HiggsAudioV3 vllm-omni launcher %s: %s",
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
                    "HiggsAudioV3 vllm-omni startup timeout after 1800 seconds"
                )

            if self.process.poll() is not None:
                stderr_tail = self._drain_stderr_tail()
                raise RuntimeError(
                    "HiggsAudioV3 vllm-omni launcher exited early with code "
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
                            "HiggsAudioV3 vllm-omni is ready at %s",
                            self.speech_url,
                        )
                        return
                    logger.info(line)
                elif stream == self.process.stderr:
                    err = self.process.stderr.readline().strip()
                    if err:
                        logger.info("HiggsAudioV3 launcher stderr: %s", err)

    def wait_until_ready(self):
        """Public hook used by ``IsolatedModelPool`` to pre-warm the instance."""
        self._ensure_server_ready()

    def _build_request_payload(self, payload: Dict):
        text = payload.pop("text", "").strip()
        if not text:
            raise ValueError("HiggsAudio v3 TTS requires a non-empty `text` field.")

        prompt_audio = payload.pop("prompt_audio", None) or payload.pop(
            "reference_audio", None
        )
        prompt_text = payload.pop("prompt_text", None) or payload.pop(
            "reference_text", None
        )
        # ``language`` / ``voice`` are informational only for v3.
        payload.pop("language", None)
        payload.pop("voice", None)
        # legacy fields not supported by /v1/audio/speech
        payload.pop("scene_prompt", None)
        payload.pop("temperature", None)
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        payload.pop("stop", None)

        response_format = str(payload.pop("response_format", "wav")).lower()
        max_new_tokens = int(payload.pop("max_new_tokens", self.default_max_new_tokens))
        seed = int(payload.pop("seed", self.default_seed))

        request_payload: Dict = {
            "model": self._model_id,
            "input": text,
            "response_format": response_format,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        }
        if prompt_audio:
            if not os.path.exists(prompt_audio):
                raise FileNotFoundError(
                    f"HiggsAudio v3 reference audio not found: {prompt_audio}"
                )
            request_payload["ref_audio"] = _audio_data_url(prompt_audio)
            if prompt_text:
                request_payload["ref_text"] = prompt_text

        # Forward any leftover knobs for forward-compat (e.g. user-supplied
        # vllm-omni-specific extras passed through ``sample_params``).
        for key, value in payload.items():
            request_payload.setdefault(key, value)
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
                "HiggsAudioV3 vllm-omni request failed: "
                f"status={response.status_code}, body={response.text[:500]}"
            )

        # /v1/audio/speech returns the raw audio bytes (not JSON).
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
