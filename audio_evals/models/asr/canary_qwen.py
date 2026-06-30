"""NVIDIA Canary-Qwen-2.5B (SALM) model wrapper for UltraEval-Audio.

Canary-Qwen-2.5B is a Speech-Augmented Language Model (SALM) that combines
a Canary speech encoder with a Qwen LLM decoder.

The actual model runs in an isolated subprocess (managed by the
``@isolated`` decorator) so its heavy dependencies do not pollute the main
UltraEval-Audio environment.

Reference:
    https://huggingface.co/nvidia/canary-qwen-2.5b
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


@isolated("audio_evals/lib/Canary-Qwen/asr_main.py")
class CanaryQwen(OfflineModel):
    """NVIDIA Canary-Qwen-2.5B (SALM) wrapper."""

    def __init__(
        self,
        path: str = "init_model/nvidia/canary-qwen-2.5b",
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        prompt_text: str = "Transcribe the following:",
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            try:
                path = self._download_model(path)
            except Exception as e:
                logger.warning(
                    "Canary-Qwen path %s does not exist locally; the "
                    "subprocess will attempt HuggingFace cache resolution. "
                    "Original error: %s",
                    path, e,
                )

        self.command_args = {
            "path": path,
            "dtype": dtype,
            "device": device,
            "max_new_tokens": str(max_new_tokens),
            "prompt_text": prompt_text,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _process_prompt(self, prompt: PromptStruct) -> Dict[str, str]:
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
            f"Cannot find audio path in prompt for Canary-Qwen: {prompt}"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)

        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "CanaryQwen subprocess has exited with code "
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
                logger.debug("CanaryQwen prompt written to stdin")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            if not rlist and self.process.poll() is not None:
                raise RuntimeError(
                    "CanaryQwen subprocess exited unexpectedly with code "
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
                            raise RuntimeError(f"CanaryQwen failed: {result}")
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
                                "[NeMo",
                                "NeMo I",
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
                                "RuntimeWarning",
                                "deprecated",
                                "pkg_resources",
                                "warnings.warn",
                            ]
                        ):
                            logger.warning(f"Process stderr: {err}")
                        else:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {e}")
