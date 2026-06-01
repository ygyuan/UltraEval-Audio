"""MiMo-V2.5-ASR model wrapper for UltraEval-Audio.

Wraps the MiMo-V2.5-ASR ASR model (XiaomiMiMo/MiMo-V2.5-ASR) so it can be
used as an offline ASR model in the UltraEval-Audio framework.

The actual model runs in an isolated subprocess (managed by the
``@isolated`` decorator) so that its heavy / version-specific dependencies
(transformers 4.49.0, torch 2.6.0, triton 3.2.0, ...) do not pollute the
main UltraEval-Audio environment.

Reference:
    https://github.com/XiaomiMiMo/MiMo-V2.5-ASR
    third_party/MiMo-V2.5-ASR/run_mimo_asr.py
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


@isolated("audio_evals/lib/MiMo-V2.5-ASR/asr_main.py")
class MiMoASR(OfflineModel):
    """MiMo-V2.5-ASR model wrapper (XiaomiMiMo/MiMo-V2.5-ASR)."""

    def __init__(
        self,
        path: str = "init_model/XiaomiMiMo/MiMo-V2.5-ASR",
        tokenizer_path: str = "init_model/XiaomiMiMo/MiMo-Audio-Tokenizer",
        language: Optional[str] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            path: Path or HuggingFace ID of the MiMo-V2.5-ASR checkpoint.
            tokenizer_path: Path or HuggingFace ID of MiMo-Audio-Tokenizer.
            language: Target transcription language (``"zh"`` / ``"en"`` /
                ``""``/``"auto"``).  When set, the corresponding ``audio_tag``
                (``<chinese>`` / ``<english>``) is forwarded to
                ``MimoAudio.asr_sft`` to constrain the output language.
            sample_params: Optional sampling parameters forwarded to the
                subprocess at inference time.
        """
        if not os.path.exists(path):
            try:
                path = self._download_model(path)
            except Exception as e:
                logger.warning(
                    "MiMo-V2.5-ASR path %s does not exist locally; the "
                    "subprocess will attempt HuggingFace cache resolution. "
                    "Original error: %s",
                    path, e,
                )
        if not os.path.exists(tokenizer_path):
            try:
                tokenizer_path = self._download_model(tokenizer_path)
            except Exception as e:
                logger.warning(
                    "MiMo-Audio-Tokenizer path %s does not exist locally; "
                    "the subprocess will attempt HuggingFace cache "
                    "resolution. Original error: %s",
                    tokenizer_path, e,
                )

        self.command_args = {
            "path": path,
            "tokenizer_path": tokenizer_path,
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
            f"Cannot find audio path in prompt for MiMo-V2.5-ASR: {prompt}"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)

        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "MiMoASR subprocess has exited with code "
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
                logger.debug("MiMoASR prompt written to stdin")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            if not rlist and self.process.poll() is not None:
                raise RuntimeError(
                    "MiMoASR subprocess exited unexpectedly with code "
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
                                f"MiMoASR failed: {result}"
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
