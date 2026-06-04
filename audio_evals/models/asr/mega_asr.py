"""Mega-ASR model wrapper for UltraEval-Audio.

Wraps the Mega-ASR model (Tsinghua / zhifeixie team, built on top of
Qwen3-ASR-1.7B) so it can be used as an offline ASR model in the
UltraEval-Audio framework.

The actual model runs in an isolated subprocess (managed by the
``@isolated`` decorator) so that its heavy / version-specific dependencies
(transformers==4.57.6, torch==2.6.0, qwen-asr, peft, ...) do not pollute
the main UltraEval-Audio environment.

Reference:
    https://github.com/xzf-thu/Mega-ASR
    third_party/Mega-ASR/infer.py
    third_party/Mega-ASR/src/MegaASR/model/megaASR.py
    init_model/zhifeixie/Mega-ASR/README.md
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


@isolated("audio_evals/lib/Mega-ASR/asr_main.py")
class MegaASR(OfflineModel):
    """Mega-ASR model wrapper (zhifeixie/Mega-ASR)."""

    def __init__(
        self,
        ckpt_dir: str = "zhifeixie/Mega-ASR",
        model_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        router_checkpoint: Optional[str] = None,
        routing_enabled: bool = True,
        quality_threshold: float = 0.5,
        dtype: str = "bfloat16",
        device_map: str = "cuda:0",
        attn_implementation: str = "auto",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        keep_delta_on_gpu: bool = True,
        backend: str = "transformers",
        language: Optional[str] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            ckpt_dir: Mega-ASR checkpoint root directory containing
                ``Qwen3-ASR-1.7B/``, ``mega-asr-merged/`` and
                ``audio_quality_router/best_acc_model.safetensors``.
                When ``model_path`` / ``lora_dir`` / ``router_checkpoint``
                are not explicitly provided, they are derived from this.
            model_path: Override path of the Qwen3-ASR-1.7B backbone.
            lora_dir: Override path of the Mega-ASR LoRA / merged adapter.
            router_checkpoint: Override path of the audio quality router.
            routing_enabled: If True, the router decides whether the LoRA
                path or base path is used per audio.
            quality_threshold: Router degraded probability threshold.
            dtype: ``"float16"`` / ``"bfloat16"`` / ``"float32"``.
            device_map: e.g. ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``.
            attn_implementation: ``"flash_attention_2"`` / ``"sdpa"`` /
                ``"eager"`` / ``"auto"``.
            max_inference_batch_size: Inference batch size cap forwarded to
                ``Qwen3ASRModel.from_pretrained``.
            max_new_tokens: Maximum tokens to generate per chunk
                (Mega-ASR default is 256).
            keep_delta_on_gpu: Whether to keep the LoRA delta tensors on
                GPU (faster but uses more memory).
            backend: ``"transformers"`` (default) or ``"vllm"``.
            language: Default target transcription language.  Accepts both
                short codes (``"zh"`` / ``"en"``) and canonical
                Qwen3-ASR names (``"Chinese"`` / ``"English"`` / ...).
                When set, the prompt forces the model to output the
                transcription in this language only.
            sample_params: Optional sampling parameters forwarded to the
                subprocess at inference time.
        """
        # Resolve ckpt_dir to a local directory.  If it doesn't already
        # exist locally, treat it as a HuggingFace repo_id (e.g.
        # ``zhifeixie/Mega-ASR``) and download the entire repository --
        # this contains all three sub-folders (Qwen3-ASR-1.7B/,
        # mega-asr-merged/, audio_quality_router/).  We MUST download the
        # whole repo here rather than per-subdir, because individual
        # sub-paths like "zhifeixie/Mega-ASR/Qwen3-ASR-1.7B" are NOT valid
        # HF repo_ids (HF only allows two-segment ``namespace/name``).
        if not os.path.exists(ckpt_dir):
            try:
                ckpt_dir = self._download_model(ckpt_dir)
            except Exception as e:
                logger.warning(
                    "Mega-ASR ckpt_dir %s does not exist locally and "
                    "could not be downloaded; the subprocess will fail "
                    "if the path is invalid. Original error: %s",
                    ckpt_dir, e,
                )

        # Derive default sub-paths from the (now-resolved) ckpt_dir if not
        # explicitly given.
        if model_path is None:
            model_path = os.path.join(ckpt_dir, "Qwen3-ASR-1.7B")
        if lora_dir is None:
            lora_dir = os.path.join(ckpt_dir, "mega-asr-merged")
        if router_checkpoint is None:
            router_checkpoint = os.path.join(
                ckpt_dir,
                "audio_quality_router",
                "best_acc_model.safetensors",
            )

        self.command_args = {
            "model_path": model_path,
            "lora_dir": lora_dir,
            "router_checkpoint": router_checkpoint,
            "routing_enabled": "true" if routing_enabled else "false",
            "quality_threshold": str(quality_threshold),
            "dtype": dtype,
            "device_map": device_map,
            "attn_implementation": attn_implementation,
            "max_inference_batch_size": str(max_inference_batch_size),
            "max_new_tokens": str(max_new_tokens),
            "keep_delta_on_gpu": "true" if keep_delta_on_gpu else "false",
            "backend": backend,
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
            f"Cannot find audio path in prompt for Mega-ASR: {prompt}"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)

        # Make sure the subprocess is alive (auto-restart on crash).
        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "MegaASR subprocess has exited with code "
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
                logger.debug("MegaASR prompt written to stdin")
                break

        # Read response.
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            if not rlist and self.process.poll() is not None:
                raise RuntimeError(
                    "MegaASR subprocess exited unexpectedly with code "
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
                                f"MegaASR failed: {result}"
                            )
                        else:
                            logger.info(result)
                    elif stream is self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if not err:
                            continue
                        # Continuation lines of a Python warning emitted by
                        # ``warnings.warn`` are split across multiple stderr
                        # lines, e.g.:
                        #   ``...transformer.py:385: UserWarning: enable_nested_tensor ...``  <- line 1 (has "UserWarning")
                        #   ``  warnings.warn(``                                              <- line 2 (no keyword)
                        # We track the previous classified level so such
                        # continuation lines inherit the WARNING level instead
                        # of being misclassified as ERROR.
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
                                "Setting `pad_token_id`",
                                "pad_token_id",
                                "generation flags are not valid",
                                "may be ignored",
                                "TRANSFORMERS_VERBOSITY",
                                "for open-end generation",
                            ]
                        ):
                            logger.debug(f"Process stderr: {err}")
                            self._last_stderr_level = "debug"
                        elif any(
                            kw in err
                            for kw in [
                                "WARNING",
                                "FutureWarning",
                                "UserWarning",
                                "DeprecationWarning",
                                "RuntimeWarning",
                                "PendingDeprecationWarning",
                                "deprecated",
                                "pkg_resources",
                                "attention_mask",
                                "pad token",
                                # Common torch.nn TransformerEncoder warning
                                # whose first line carries "UserWarning" and
                                # whose continuation lines we want to keep
                                # at WARNING level.
                                "enable_nested_tensor",
                                "use_nested_tensor",
                                "encoder_layer.norm_first",
                                # Generic Python ``warnings.warn(...)`` /
                                # ``warnings.warn_explicit(...)`` callsite
                                # echo printed by warnings module.
                                "warnings.warn",
                            ]
                        ):
                            logger.warning(f"Process stderr: {err}")
                            self._last_stderr_level = "warning"
                        elif getattr(self, "_last_stderr_level", None) in (
                            "warning",
                            "debug",
                        ):
                            # Inherit the previous level for unclassified
                            # continuation lines (typical for multi-line
                            # Python warnings / tracebacks-of-warnings).
                            level_method = getattr(
                                logger,
                                self._last_stderr_level,
                                logger.warning,
                            )
                            level_method(f"Process stderr: {err}")
                        else:
                            logger.error(f"Process stderr: {err}")
                            self._last_stderr_level = "error"
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {e}")
