"""GLM-4-Voice model client for UltraEval-Audio.

This client talks to the FastAPI adapter server defined in
``third_party/GLM-4-Voice/ultraeval_adapter_server.py`` (``/generate_stream``).

The adapter expects ``{"prompt": <task instruction text>, "audio": <wav b64>}``
and returns a JSON line with ``{"text": str, "audio": <float-list>, "sampleRate": int}``.

For text-only tasks (ASR / S2TT / emotion) we just return ``text``.

For S2S spoken-QA tasks (where the post-process pipeline relies on
``extract_audio`` + ``speech2text``), we persist the returned waveform to a
unique ``.wav`` file under ``$AUDIO_EVALS_OUTPUT_DIR/glm4voice_audio`` (falling
back to the system tmp dir) and return a JSON string ``{"text": ..., "audio":
"/abs/path.wav"}`` so the standard ``extract_audio`` post-processor can pick
it up downstream.
"""

import json
import logging
import os
import subprocess
import tempfile
import uuid
from typing import Dict

import numpy as np
import requests
import soundfile as sf

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file


logger = logging.getLogger(__name__)


# Tasks whose downstream post-processing requires a real wav file
# (extract_audio + speech2text).  When the task instruction is empty (audio
# only prompt template) we also assume S2S because the upstream prompt for
# spoken-QA in registry/prompt/glm4voice.yaml has no text content.
_S2S_KEYWORDS = (
    "speech-qa",
    "spoken qa",
    "spoken-qa",
    "speech qa",
    "audio-only",
)


def _looks_like_s2s(text_prompt: str) -> bool:
    """Heuristic: empty prompt -> audio-only (spoken QA / choice) task."""
    if not text_prompt or not text_prompt.strip():
        return True
    lowered = text_prompt.lower()
    return any(k in lowered for k in _S2S_KEYWORDS)


def _ensure_wav(audio_file: str) -> tuple[str, bool]:
    """Make sure the input is a 24kHz wav file expected by the GLM-4-Voice
    speech tokenizer.  Returns (path, is_temp)."""
    _, ext = os.path.splitext(audio_file)
    if ext.lower() == ".wav":
        return audio_file, False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_file, "-ar", "24000", tmp.name],
        capture_output=True,
        text=True,
        check=True,
    )
    return tmp.name, True


def _save_waveform(waveform: np.ndarray, sample_rate: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"glm4voice_{uuid.uuid4().hex}.wav")
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    sf.write(path, waveform, sample_rate)
    return path


def _parse_streaming_response(response: requests.Response) -> dict:
    """The adapter writes one JSON object terminated by ``\0``."""
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            return json.loads(chunk.decode("utf-8"))
    raise RuntimeError("Empty response from GLM-4-Voice adapter server")


class GLM4Voice(APIModel):
    """Client for the GLM-4-Voice ultraeval adapter server."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:10001/generate_stream",
        sample_params: Dict[str, any] = None,
        audio_out_dir: str = None,
        request_timeout: int = 1800,
    ):
        super().__init__(True, sample_params)
        self.url = url
        self.request_timeout = request_timeout
        if audio_out_dir is None:
            audio_out_dir = os.environ.get(
                "GLM4VOICE_AUDIO_OUT_DIR",
                os.path.join(tempfile.gettempdir(), "glm4voice_audio"),
            )
        self.audio_out_dir = audio_out_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_audio_and_text(prompt: PromptStruct) -> tuple[str, str]:
        audio_file, text_prompt = "", ""
        for content in prompt:
            if content.get("role") != "user":
                continue
            for line in content.get("contents", []):
                if line.get("type") == "audio" and not audio_file:
                    audio_file = line.get("value", "")
                elif line.get("type") == "text":
                    if text_prompt:
                        text_prompt = text_prompt + "\n" + line.get("value", "")
                    else:
                        text_prompt = line.get("value", "")
        return audio_file, text_prompt

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_file, text_prompt = self._extract_audio_and_text(prompt)
        if not audio_file:
            raise ValueError("GLM-4-Voice prompt must contain an audio entry")

        wav_path, is_temp = _ensure_wav(audio_file)
        try:
            audio_b64 = get_base64_from_file(wav_path)
        finally:
            if is_temp and os.path.exists(wav_path):
                os.remove(wav_path)

        # ``kwargs`` already merges ``self.sample_params`` with any per-call
        # overrides, see ``Model.inference``.
        payload = {
            "prompt": text_prompt or "",
            "audio": audio_b64,
            "temperature": float(kwargs.get("temperature", 0.2)),
            "top_p": float(kwargs.get("top_p", 0.8)),
            "max_new_tokens": int(kwargs.get("max_new_tokens", 2000)),
        }

        response = requests.post(
            self.url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GLM-4-Voice server {self.url} returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = _parse_streaming_response(response)
        text = (data.get("text") or "").strip()

        if _looks_like_s2s(text_prompt):
            waveform = data.get("audio") or []
            sample_rate = int(data.get("sampleRate") or 22050)
            if len(waveform) <= 1:
                # Adapter returned a placeholder (no real audio); still emit a
                # JSON dict so downstream extract_audio post-processing does
                # not crash.  speech2text on a near-empty wav simply returns
                # an empty string.
                logger.warning(
                    "GLM-4-Voice S2S response has no audio; writing 1-sample silent wav."
                )
            wav_out = _save_waveform(np.asarray(waveform, dtype=np.float32), sample_rate, self.audio_out_dir)
            return json.dumps({"text": text, "audio": wav_out}, ensure_ascii=False)

        return text
