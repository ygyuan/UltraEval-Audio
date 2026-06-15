"""Isolated subprocess entry for the dots.tts-soar (and sibling) TTS models.

Reference inference code: third_party/dots.tts/src/dots_tts/runtime.py
                          third_party/dots.tts/src/dots_tts/cli.py

Communication protocol (shared with other UltraEval-Audio TTS subprocesses):
- stdin lines look like ``"<uuid>->{json_payload}\n"``.
- json payload supports the standard ``voice-clone`` prompt template:
      {"text": "...", "prompt_audio": "/abs/path.wav", "prompt_text": "..."}
  Optional extra keys (forwarded as runtime.generate kwargs when present):
      language, template_name, ode_method, num_steps, guidance_scale,
      speaker_scale, normalize_text, seed.
- stdout response: ``"<uuid>-><wav_path_or_json>\n"``.
"""
import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import soundfile as sf

# dots.tts is shipped as a third_party source tree. Add ``src`` to PYTHONPATH
# so ``import dots_tts`` works without a separate ``pip install -e .`` step
# inside the isolated venv.
_PROJECT_ROOT = os.path.abspath(os.getcwd())
_DOTS_TTS_SRC = os.path.join(_PROJECT_ROOT, "third_party", "dots.tts", "src")
if _DOTS_TTS_SRC not in sys.path:
    sys.path.insert(0, _DOTS_TTS_SRC)

# Suppress noisy third-party warnings that flood stderr (one line per worker)
# and make the real traceback unreadable when multiple workers run in parallel.
import warnings  # noqa: E402

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.nn\.utils\.weight_norm.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*CUDA initialization: The NVIDIA driver on your system.*",
)

import torch  # noqa: E402

from dots_tts.runtime import DotsTtsRuntime  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Generation kwargs forwarded to ``DotsTtsRuntime.generate`` when present in the
# per-request payload. Anything else in the payload is silently ignored.
_GENERATE_KEYS = (
    "language",
    "template_name",
    "ode_method",
    "num_steps",
    "guidance_scale",
    "speaker_scale",
    "normalize_text",
    "profile_inference",
)


def _install_xvec_proj_dtype_bridge(runtime: DotsTtsRuntime) -> None:
    """Cast fp32 speaker embeddings to the dtype expected by ``core.xvec_proj``.

    The upstream speaker encoder explicitly disables autocast and returns fp32
    embeddings. ``DotsTtsRuntime`` casts ``model.core`` to the requested
    inference dtype, so ``core.xvec_proj`` is bf16/fp16 when low-precision
    inference is enabled. Some PyTorch builds do not bridge this particular
    ``Sequential`` boundary automatically, causing ``mat1 and mat2 must have
    the same dtype`` during voice-cloning prompt conditioning.
    """
    xvec_proj = getattr(getattr(runtime.model, "core", None), "xvec_proj", None)
    if xvec_proj is None:
        logger.warning("dots.tts xvec_proj not found; dtype bridge was not installed")
        return

    try:
        target_param = next(xvec_proj.parameters())
    except StopIteration:
        logger.warning("dots.tts xvec_proj has no parameters; dtype bridge skipped")
        return

    target_device = target_param.device
    target_dtype = target_param.dtype

    def _cast_xvec_input(_module, inputs):
        if not inputs:
            return inputs
        speaker_embedding, *rest = inputs
        if torch.is_tensor(speaker_embedding):
            speaker_embedding = speaker_embedding.to(
                device=target_device,
                dtype=target_dtype,
            )
        return (speaker_embedding, *rest)

    xvec_proj.register_forward_pre_hook(_cast_xvec_input)
    logger.info(
        "Installed dots.tts xvec_proj dtype bridge: speaker_embedding -> %s on %s",
        target_dtype,
        target_device,
    )


def _build_runtime(args) -> DotsTtsRuntime:
    logger.info(
        "Loading dots.tts model: path=%s precision=%s optimize=%s "
        "max_generate_length=%s",
        args.path,
        args.precision,
        args.optimize,
        args.max_generate_length,
    )

    # ------------------------------------------------------------------
    # Strategy: stay on dots.tts's official inference path while patching only
    # the one boundary that fails in UltraEval voice-cloning prompts.
    #
    # Upstream ``DotsTtsRuntime`` casts only ``model.core`` to the requested
    # low-precision dtype and keeps the speaker encoder / vocoder in fp32.
    # That is mostly correct and avoids quality regressions, but the speaker
    # encoder also disables autocast and returns fp32 embeddings. On our stack,
    # ``core.xvec_proj`` receives those embeddings directly after ``core`` has
    # been cast to bf16/fp16, producing:
    #     RuntimeError: mat1 and mat2 must have the same dtype
    #
    # Do NOT install a global dtype hook or cast the full model: those broader
    # fixes interfere with kv-cache/vocoder fp32 paths. Instead, install a
    # narrow pre-hook on ``core.xvec_proj`` after runtime initialization.
    # ------------------------------------------------------------------
    runtime = DotsTtsRuntime.from_pretrained(
        args.path,
        precision=args.precision,
        optimize=bool(args.optimize),
        max_generate_length=args.max_generate_length,
    )
    _install_xvec_proj_dtype_bridge(runtime)

    logger.info(
        "dots.tts model loaded successfully (sample_rate=%d)", runtime.sample_rate
    )
    return runtime


def _resolve_generate_kwargs(payload: dict, defaults: dict) -> dict:
    """Build kwargs for ``runtime.generate`` from request payload + CLI defaults."""
    kwargs = dict(defaults)
    for key in _GENERATE_KEYS:
        if key in payload:
            kwargs[key] = payload[key]
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Local pretrained directory or HuggingFace repo id for dots.tts",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help=(
            "Inference precision. ``bfloat16`` is recommended on A100/H100 "
            "(the official dots.tts CLI uses bfloat16 by default). The "
            "upstream runtime casts only ``model.core`` to the target dtype; "
            "this wrapper additionally bridges the fp32 speaker embedding "
            "into ``core.xvec_proj`` for voice-cloning prompts."
        ),
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable torch.compile acceleration (slower warmup, faster steady-state)",
    )
    parser.add_argument(
        "--max_generate_length",
        type=int,
        default=500,
        help="Maximum total audio patch count (prompt + generated)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Default flow-matching sampling steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.2,
        help="Default classifier-free guidance scale",
    )
    parser.add_argument(
        "--speaker_scale",
        type=float,
        default=1.5,
        help="Default speaker embedding scale",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default="euler",
        help="Default ODE solver method",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="tts",
        help="Default generation template name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Default language tag (e.g. EN/ZH/auto_detect/none)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic generation",
    )
    parser.add_argument(
        "--normalize_text",
        action="store_true",
        help="Apply WeTextProcessing normalization before inference",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    runtime = _build_runtime(args)

    default_generate_kwargs = {
        "template_name": args.template_name,
        "ode_method": args.ode_method,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "speaker_scale": args.speaker_scale,
        "normalize_text": args.normalize_text,
    }
    if args.language:
        default_generate_kwargs["language"] = args.language

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info("ENABLE_RTF: %d", enable_rtf)

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain '->', but got {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            payload = json.loads(prompt[anchor + 2 :])

            text = payload.pop("text")
            prompt_audio = payload.pop("prompt_audio", None) or None
            prompt_text = payload.pop("prompt_text", None) or None

            generate_kwargs = _resolve_generate_kwargs(payload, default_generate_kwargs)
            logger.info(
                "dots.tts request: text_len=%d has_prompt_audio=%s "
                "has_prompt_text=%s generate_kwargs=%s",
                len(text or ""),
                bool(prompt_audio),
                bool(prompt_text),
                generate_kwargs,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            result = runtime.generate(
                text=text,
                prompt_audio_path=prompt_audio,
                prompt_text=prompt_text,
                **generate_kwargs,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_time

            audio = result["audio"].float().cpu().squeeze().numpy()
            sample_rate = int(result["sample_rate"])

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, samplerate=sample_rate)
                output_path = f.name

            if enable_rtf == 1:
                audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
                rtf = (
                    inference_time / audio_duration if audio_duration > 0 else 0.0
                )
                response = json.dumps({"audio": output_path, "RTF": rtf})
                logger.info(
                    "RTF: %.4f (inference: %.2fs, audio: %.2fs)",
                    rtf,
                    inference_time,
                    audio_duration,
                )
            else:
                response = output_path

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{response}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:  # noqa: BLE001
            import traceback

            # Emit the full traceback as a SINGLE stderr write so the parent
            # process does not interleave lines from multiple concurrent
            # workers (which makes logs unreadable, see app-2026-06-12 logs).
            tb = traceback.format_exc()
            sys.stderr.write(tb if tb.endswith("\n") else tb + "\n")
            sys.stderr.flush()
            print(f"Error: {str(e)}", flush=True)
