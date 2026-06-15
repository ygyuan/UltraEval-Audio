"""dots.tts model wrapper for UltraEval-Audio.

Wraps the rednote-hilab dots.tts family (`dots.tts-base`, `dots.tts-soar`,
`dots.tts-mf`) behind UltraEval-Audio's isolated-subprocess `OfflineModel`
contract. The actual inference runs out-of-process via
``audio_evals/lib/DotsTTS/main.py``.

Reference: third_party/dots.tts/src/dots_tts/runtime.py
"""
import json
import logging
import os
import select
from typing import Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/DotsTTS/main.py")
class DotsTTS(OfflineModel):
    """dots.tts (rednote-hilab) zero-shot voice-cloning TTS.

    The standard ``voice-clone`` prompt template emits::

        {"text": "...", "prompt_audio": "...", "prompt_text": "..."}

    which is exactly the payload schema the subprocess understands.
    """

    def __init__(
        self,
        path: str,
        precision: str = "float32",
        optimize: bool = False,
        max_generate_length: int = 500,
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        template_name: str = "tts",
        language: Optional[str] = None,
        normalize_text: bool = False,
        seed: int = 42,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """Construct a DotsTTS isolated runner.

        Args:
            path: Local model directory or HuggingFace repo id (e.g.
                ``init_model/rednote-hilab/dots.tts-soar`` or
                ``rednote-hilab/dots.tts-soar``).
            precision: Inference precision (``float32``/``bfloat16``/``float16``).
                Defaults to ``float32``. dots.tts ``runtime.py`` only casts
                ``model.core`` to the requested dtype while keeping the speaker
                encoder and vocoder in fp32. Setting ``bfloat16`` exposes
                multiple fp32->bf16 boundaries inside ``core`` (xvec_proj /
                several internal Conv/Linear layers) that we cannot patch
                exhaustively from the wrapper, so we ship the wrapper at fp32
                by default and rely on multi-worker parallelism for speedup.
                See ``registry/model/dots_tts.yaml`` for the full rationale.
            optimize: Enable ``torch.compile`` warmup for faster steady-state.
            max_generate_length: Max audio patch count (prompt + generated).
            num_steps: Default flow-matching sampling steps.
            guidance_scale: Default classifier-free guidance scale.
            speaker_scale: Default reference-speaker embedding scale.
            ode_method: ODE solver method (``euler`` etc.).
            template_name: Generation template (``tts``/``instruction_tts``/...).
            language: Optional language tag forwarded to runtime
                (``EN``/``ZH``/``auto_detect``/``none``/...).
            normalize_text: Apply WeTextProcessing normalization.
            seed: RNG seed for deterministic sampling.
            sample_params: Extra sampling overrides forwarded per-request.
        """
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "precision": precision,
            "max_generate_length": str(max_generate_length),
            "num_steps": str(num_steps),
            "guidance_scale": str(guidance_scale),
            "speaker_scale": str(speaker_scale),
            "ode_method": ode_method,
            "template_name": template_name,
            "seed": str(seed),
        }
        if optimize:
            self.command_args["optimize"] = ""
        if normalize_text:
            self.command_args["normalize_text"] = ""
        if language:
            self.command_args["language"] = language

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        if isinstance(prompt, dict):
            prompt.update(kwargs)
        else:
            prompt = {"text": prompt, **kwargs}

        # Defensive: strip empty strings so the subprocess treats them as None
        # (e.g. the ``voice-clone`` template renders ``prompt_text: ""`` when
        # the dataset row has no transcript -- dots.tts requires either both
        # prompt_audio+prompt_text or neither).
        if prompt.get("prompt_text") == "":
            prompt.pop("prompt_text", None)
        if prompt.get("prompt_audio") == "":
            prompt.pop("prompt_audio", None)

        # Detect a dead worker subprocess up front so we can fail this single
        # sample fast (and let the eval runner mark it failed and continue)
        # instead of blocking ``select.select`` for 600s on a closed stdin /
        # crashing with ``ValueError: I/O operation on closed file`` deep in
        # the read loop.
        if self.process.poll() is not None or self.process.stdin is None or self.process.stdin.closed:
            raise RuntimeError(
                f"dots.tts worker subprocess is not alive (returncode={self.process.poll()}); "
                f"cannot serve this sample"
            )

        try:
            while True:
                _, wlist, _ = select.select([], [self.process.stdin], [], 600)
                if wlist:
                    self.process.stdin.write(
                        f"{prefix}{json.dumps(prompt, ensure_ascii=False)}\n"
                    )
                    self.process.stdin.flush()
                    logger.debug("prompt written to dots.tts stdin")
                    break
        except (ValueError, OSError, BrokenPipeError) as e:
            # The worker subprocess crashed (e.g. dtype mismatch in core,
            # CUDA OOM, etc.) and its stdin pipe was closed. Surface a
            # recoverable RuntimeError so the eval task can mark THIS sample
            # as failed and move on to the next one rather than aborting.
            raise RuntimeError(
                f"dots.tts worker stdin closed before request was sent "
                f"(worker returncode={self.process.poll()}): {e}"
            ) from e

        while True:
            try:
                rlist, _, _ = select.select(
                    [self.process.stdout, self.process.stderr], [], [], 600
                )
            except (ValueError, OSError) as e:
                # stdout/stderr pipes were closed because the worker died.
                raise RuntimeError(
                    f"dots.tts worker output pipe closed unexpectedly "
                    f"(worker returncode={self.process.poll()}): {e}"
                ) from e

            if not rlist:
                # If we hit the read timeout but the worker is also dead,
                # report the death rather than a generic timeout so the log
                # is actionable.
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"dots.tts worker exited unexpectedly with returncode="
                        f"{self.process.poll()} while waiting for response"
                    )
                err_msg = "Read timeout after 600 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            return result[len(prefix):]
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"dots.tts failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            if any(kw in err for kw in [
                                "Traceback",
                                "RuntimeError",
                                "Exception",
                                "CUDA error",
                                "OutOfMemory",
                                "mat1 and mat2",
                                "Generation failed",
                                "ERROR",
                            ]):
                                logger.error(f"Process stderr: {err}")
                            elif any(kw in err for kw in [
                                "WARNING",
                                "FutureWarning",
                                "UserWarning",
                                "DeprecationWarning",
                                "deprecated",
                                "pkg_resources",
                                "NVIDIA driver",
                            ]):
                                logger.warning(f"Process stderr: {err}")
                            elif any(kw in err for kw in [
                                "INFO",
                                "DEBUG",
                                "Loading",
                                "Loaded",
                                "loading",
                                "loaded",
                                "Building",
                                "building",
                                "done",
                                "Runtime",
                                "%|",
                                "it/s]",
                            ]):
                                logger.debug(f"Process stderr: {err}")
                            else:
                                logger.info(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
