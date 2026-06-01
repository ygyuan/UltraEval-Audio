"""
GLM-ASR model wrapper for UltraEval-Audio.

Wraps GLM-ASR-Nano (and other GLM-ASR variants from zai-org) so it can be
used as an offline ASR model in the UltraEval-Audio framework.

The actual model runs in an isolated subprocess (managed by the
``@isolated`` decorator) so that its heavy / version-specific dependencies
(transformers 5.x, torchcodec, ...) do not pollute the main UltraEval-Audio
environment.

Reference:
    https://github.com/zai-org/GLM-ASR
    init_model/zai-org/GLM-ASR-Nano-2512/README.md
"""

import json
import logging
import os
import select
import sys
import uuid
from typing import Any, Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/GLM_ASR/asr_main.py")
class GLMASR(OfflineModel):
    """GLM-ASR model wrapper (e.g. zai-org/GLM-ASR-Nano-2512)."""

    def __init__(
        self,
        path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        attn_implementation: str = "auto",
        max_new_tokens: int = 500,
        language: Optional[str] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            path: Path or HuggingFace ID of the GLM-ASR checkpoint.
            dtype: ``"float16"`` / ``"bfloat16"`` / ``"float32"``.
            device: e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``.
            attn_implementation: ``"flash_attention_2"`` / ``"sdpa"`` /
                ``"eager"`` / ``"auto"``.
            max_new_tokens: Default max tokens for each generation.
            language: Target transcription language (``"zh"`` / ``"en"``).
                When set, a language-specific prompt is sent to the model
                to nudge the output language.
            sample_params: Optional sampling parameters forwarded to the
                subprocess at inference time.
        """
        if not os.path.exists(path):
            # Allow passing a HF repo id; the subprocess will try to
            # resolve it through the HuggingFace cache.
            logger.warning(
                "GLM-ASR path %s does not exist locally; the subprocess "
                "will attempt HuggingFace cache resolution.",
                path,
            )

        self.command_args = {
            "path": path,
            "dtype": dtype,
            "device": device,
            "attn_implementation": attn_implementation,
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
            f"Cannot find audio path in prompt for GLM-ASR: {prompt}"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)

        # Make sure the subprocess is alive (auto-restart on crash).
        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "GLMASR subprocess has exited with code "
                f"{self.process.returncode}."
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        prompt["kwargs"] = kwargs

        # Send request.
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(prompt, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                logger.debug("GLMASR prompt written to stdin")
                break

        # Read response.
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            if not rlist and self.process.poll() is not None:
                raise RuntimeError(
                    "GLMASR subprocess exited unexpectedly with code "
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
                            raise RuntimeError(
                                f"GLMASR failed: {result}"
                            )
                        else:
                            logger.info(result)
                    elif stream is self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if not err:
                            continue
                        if any(
                            kw in err
                            for kw in [
                                "INFO",
                                "DEBUG",
                                "Loading",
                                "Building",
                                "loading",
                                "building",
                                "done",
                                "loaded",
                                "%|",
                                "it/s]",
                                "not found close signal",
                                "close signal not received",
                            ]
                        ):
                            logger.debug(f"Process stderr: {err}")
                        elif any(
                            kw in err
                            for kw in [
                                "WARNING",
                                "FutureWarning",
                                "UserWarning",
                                "DeprecationWarning",
                                "deprecated",
                                "pkg_resources",
                                "attention_mask",
                                "pad token",
                            ]
                        ):
                            logger.warning(f"Process stderr: {err}")
                        else:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {e}")
