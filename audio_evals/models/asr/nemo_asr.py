"""NVIDIA NeMo ASR model wrapper for UltraEval-Audio.

Wraps NVIDIA NeMo ASR models so they can be used as offline ASR models
in the UltraEval-Audio framework.

Supported architectures (auto-detected from the checkpoint path):
    * ``parakeet-tdt-0.6b-v3``  (EncDecRNNTBPEModel — multilingual TDT)
    * ``canary-1b-v2``          (EncDecMultiTaskModel — multitask ASR+AST)

The actual model runs in an isolated subprocess (managed by the
``@isolated`` decorator) so its heavy / version-specific dependencies
(nemo_toolkit[asr], torch, pytorch-lightning, ...) do not pollute the
main UltraEval-Audio environment.

Reference:
    https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
    https://huggingface.co/nvidia/canary-1b-v2
"""

import json
import logging
import os
import select
import uuid
from typing import Any, Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/NeMo-ASR/asr_main.py")
class NemoASR(OfflineModel):
    """NVIDIA NeMo ASR model wrapper (parakeet / canary)."""

    def __init__(
        self,
        path: str = "init_model/nvidia/parakeet-tdt-0.6b-v3",
        model_class: str = "auto",
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        language: Optional[str] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            path: Local path (preferred) or HF repo id of the NeMo
                ASR checkpoint directory / ``.nemo`` file.
            model_class: ``"auto"`` / ``"multitask"`` / ``"rnnt"`` / ``"ctc"``.
                ``"auto"`` infers from the directory name: ``canary`` →
                ``multitask``; ``parakeet`` → ``rnnt``.
            dtype: ``"float16"`` / ``"bfloat16"`` / ``"float32"``.
            device: e.g. ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``.
            max_new_tokens: Maximum tokens to generate per chunk (only
                used by multitask / canary models).
            language: Default target transcription language (short code
                e.g. ``"en"`` / ``"zh"`` / ``"de"``).  Canary requires
                a language to be set — defaults to ``"en"`` when unset.
            sample_params: Optional sampling parameters forwarded to the
                subprocess at inference time.
        """
        if not os.path.exists(path):
            try:
                path = self._download_model(path)
            except Exception as e:
                logger.warning(
                    "NeMo-ASR path %s does not exist locally; the "
                    "subprocess will attempt HuggingFace cache resolution. "
                    "Original error: %s",
                    path, e,
                )

        self.command_args = {
            "path": path,
            "model_class": model_class,
            "dtype": dtype,
            "device": device,
            "max_new_tokens": str(max_new_tokens),
        }
        if language:
            self.command_args["language"] = language
        super().__init__(is_chat=True, sample_params=sample_params)

    def _process_prompt(self, prompt: PromptStruct) -> Dict[str, str]:
        """Extract the audio path from any common prompt structure."""
        if isinstance(prompt, dict):
            audio = (
                prompt.get("audio")
                or prompt.get("WavPath")
                or prompt.get("prompt_audio")
            )
            if audio:
                if not os.path.exists(audio):
                    raise FileNotFoundError(f"Audio file not found: {audio}")
                return {"audio": audio}
        if isinstance(prompt, list):
            for content in prompt:
                if not isinstance(content, dict):
                    continue
                for line in content.get("contents", []):
                    if line.get("type") == "audio":
                        audio = line.get("value")
                        if not os.path.exists(audio):
                            raise FileNotFoundError(
                                f"Audio file not found: {audio}"
                            )
                        return {"audio": audio}
        raise ValueError(
            f"Cannot find audio path in prompt for NemoASR: {prompt}"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)

        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "NemoASR subprocess has exited with code "
                f"{self.process.returncode}."
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        prompt["kwargs"] = kwargs

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(prompt, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                logger.debug("NemoASR prompt written to stdin")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            if not rlist and self.process.poll() is not None:
                raise RuntimeError(
                    "NemoASR subprocess exited unexpectedly with code "
                    f"{self.process.returncode}."
                )

            try:
                for stream in rlist:
                    if stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            payload = result[len(prefix):]
                            try:
                                obj = json.loads(payload)
                                return obj.get("content", payload)
                            except json.JSONDecodeError:
                                return payload
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"NemoASR failed: {result}")
                        else:
                            logger.info(result)
                    elif stream is self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if not err:
                            continue
                        # Order matters: check ERROR-ish keywords FIRST so
                        # that lines like "RuntimeError: ..." are not later
                        # swallowed by INFO/config heuristics.
                        if any(
                            kw in err
                            for kw in [
                                "Traceback (most recent call last):",
                                "Error:",
                                "ERROR",
                                "CRITICAL",
                                "FATAL",
                                "Segmentation fault",
                                "[NeMo E",
                                "NeMo E ",
                                "Exception:",
                                "RuntimeError",
                                "ValueError",
                                "TypeError",
                                "KeyError",
                                "AttributeError",
                                "ImportError",
                                "ModuleNotFoundError",
                                "FileNotFoundError",
                                "OSError",
                                "AssertionError",
                                "OutOfMemoryError",
                                "CUDA error",
                                "CUDA out of memory",
                                "killed",
                            ]
                        ):
                            logger.error(f"Process stderr: {err}")
                        elif any(
                            kw in err
                            for kw in [
                                "WARNING",
                                "FutureWarning",
                                "UserWarning",
                                "DeprecationWarning",
                                "RuntimeWarning",
                                "deprecated",
                                "pkg_resources",
                                "warnings.warn",
                                "[NeMo W",
                                "NeMo W ",
                            ]
                        ):
                            logger.warning(f"Process stderr: {err}")
                        else:
                            # Default everything else (NeMo INFO, train/val
                            # config dumps, tqdm progress bars, bashrc echoes,
                            # node version banners, ...) to INFO so the log
                            # is not flooded with bogus ERROR entries.
                            logger.info(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {e}")
