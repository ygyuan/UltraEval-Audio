"""Subprocess entry point for Mega-ASR.

This file is launched by ``audio_evals.models.asr.mega_asr.MegaASR`` (via
the ``@isolated`` decorator) inside an isolated virtualenv that has the
heavy Mega-ASR / Qwen3-ASR dependencies installed (transformers==4.57.6,
torch==2.6.0, qwen-asr, peft, ...).  It speaks the same line-based JSON
protocol as the other ASR wrappers (e.g. Qwen3-ASR / MiMo-V2.5-ASR /
VibeVoice-ASR):

    request   :  ``<uuid>-> {"audio": "/path/to.wav", "kwargs": {...}}\n``
    response  :  ``<uuid>-> {"content": "...", "raw_text": "..."}\n``
    close     :  ``<uuid>-> close\n``

The actual model code lives in the official Mega-ASR repo
(``third_party/Mega-ASR``).  The repo is NOT a pip-installable package,
so we add ``third_party/Mega-ASR/src`` to ``sys.path`` before importing
``MegaASR.model.megaASR``.

Reference:
    https://github.com/xzf-thu/Mega-ASR
    third_party/Mega-ASR/infer.py
    third_party/Mega-ASR/src/MegaASR/model/megaASR.py
"""

import argparse
import json
import logging
import os
import re
import select
import sys
import time

# Force fully offline mode for the isolated subprocess: the ASR machine
# typically has no network access, and we provide all model files locally.
# These env vars must be set BEFORE importing transformers / huggingface_hub
# so that they take effect during import-time HTTP calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Add the Mega-ASR repo's ``src`` to sys.path so that
# ``from MegaASR.model.megaASR import MegaASR`` works (this mirrors what
# ``third_party/Mega-ASR/infer.py`` does at the top of the file).
_MEGA_ASR_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "third_party", "Mega-ASR", "src")
)
if os.path.isdir(_MEGA_ASR_SRC) and _MEGA_ASR_SRC not in sys.path:
    sys.path.insert(0, _MEGA_ASR_SRC)

import torch  # noqa: E402


logger = logging.getLogger("mega-asr")
logging.basicConfig(
    level=os.environ.get("MEGA_ASR_LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# Mega-ASR's underlying Qwen3-ASR uses canonical capitalised language names
# (see third_party/Mega-ASR/qwen-asr inference utils).  We accept the short
# codes used elsewhere in UltraEval-Audio and map them to the canonical
# form expected by ``MegaASR.infer(language=...)``.
_LANG_ALIAS = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "en-us": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "ar": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hindi": "Hindi",
}


def _normalize_language(language):
    """Map a user-provided ``language`` to the canonical Qwen3-ASR name.

    Returns ``None`` for empty / "auto" input (auto-detect).
    """
    if language is None:
        return None
    s = str(language).strip()
    if not s or s.lower() in ("auto", ""):
        return None
    canonical = _LANG_ALIAS.get(s.lower())
    if canonical is not None:
        return canonical
    return s[:1].upper() + s[1:].lower()


_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)


def _clean_text(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)
    return _LEADING_ROLE_RE.sub("", text).strip()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the Qwen3-ASR-1.7B backbone directory",
    )
    parser.add_argument(
        "--lora_dir", type=str, required=True,
        help="Path to the Mega-ASR merged LoRA adapter directory",
    )
    parser.add_argument(
        "--router_checkpoint", type=str, required=True,
        help="Path to the audio quality router checkpoint",
    )
    parser.add_argument(
        "--routing_enabled", type=str, default="true",
        help="Enable audio-quality based routing between base / LoRA paths",
    )
    parser.add_argument(
        "--quality_threshold", type=float, default=0.5,
        help="Router degraded probability threshold",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Model dtype",
    )
    parser.add_argument(
        "--device_map", type=str, default="cuda:0",
        help="Device to run the model on (e.g. cuda:0 / cpu)",
    )
    parser.add_argument(
        "--attn_implementation", type=str, default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation forwarded to AutoModel.from_pretrained",
    )
    parser.add_argument(
        "--max_inference_batch_size", type=int, default=32,
        help="Inference batch size cap for chunked long-audio decoding",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum tokens to generate per chunk",
    )
    parser.add_argument(
        "--keep_delta_on_gpu", type=str, default="true",
        help="Whether to keep LoRA delta tensors resident on GPU",
    )
    parser.add_argument(
        "--backend", type=str, default="transformers",
        choices=["transformers", "vllm"],
        help="Mega-ASR backend (transformers is the only one tested here)",
    )
    parser.add_argument(
        "--language", type=str, default="",
        help=(
            "Optional default target language (e.g. 'zh', 'en', 'Chinese', "
            "'English'). When set, the prompt will force output to be the "
            "transcription text in this language only.  Empty string means "
            "auto-detect."
        ),
    )
    args = parser.parse_args()

    routing_enabled = _str2bool(args.routing_enabled)
    keep_delta_on_gpu = _str2bool(args.keep_delta_on_gpu)
    default_language = _normalize_language(args.language)

    logger.info(
        "Loading Mega-ASR: model=%s, lora=%s, router=%s, routing=%s, "
        "thr=%.3f, dtype=%s, device=%s, attn=%s, default_language=%r",
        args.model_path, args.lora_dir, args.router_checkpoint,
        routing_enabled, args.quality_threshold, args.dtype,
        args.device_map, args.attn_implementation, default_language,
    )

    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device_map == "cpu":
        dtype = torch.float32

    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        if args.device_map.startswith("cuda") and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"
        else:
            attn_impl = "sdpa"
    logger.info("Resolved attn_implementation=%s", attn_impl)

    # Resolve absolute local paths so HuggingFace cache lookup never goes
    # online.
    def _abs(p):
        return os.path.abspath(p) if p and os.path.exists(p) else p

    model_path = _abs(args.model_path)
    lora_dir = _abs(args.lora_dir)
    router_ckpt = _abs(args.router_checkpoint)

    # Lazy import: surface any import errors only after the subprocess has
    # been launched (matches Qwen3-ASR / MiMo-V2.5-ASR / VibeVoice-ASR).
    from MegaASR.model.megaASR import MegaASR  # type: ignore

    asr = MegaASR(
        model_path=model_path,
        lora_dir=lora_dir,
        router_checkpoint=router_ckpt,
        routing_enabled=routing_enabled,
        quality_threshold=args.quality_threshold,
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        keep_delta_on_gpu=keep_delta_on_gpu,
        backend=args.backend,
        # Forward extra kwargs to the underlying Qwen3ASRModel.from_pretrained
        dtype=dtype,
        attn_implementation=attn_impl,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"Model loaded from checkpoint: {model_path}", flush=True)
    logger.info("Mega-ASR model loaded successfully")

    while True:
        try:
            prompt = input()
        except EOFError:
            break

        try:
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid request format, must contain '->' but got "
                    f"{prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            payload = json.loads(prompt[anchor + 2:])

            audio_path = payload.get("audio") or payload.get("WavPath")
            if not audio_path:
                print(f"{prefix}Error: 'audio' field is required", flush=True)
                continue
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(audio_path)
            if not os.path.exists(audio_path):
                print(
                    f"{prefix}Error: audio file not found: {audio_path}",
                    flush=True,
                )
                continue

            kwargs = payload.get("kwargs", {}) or {}
            req_language = kwargs.pop("language", None)
            language_for_call = (
                _normalize_language(req_language)
                if req_language is not None
                else default_language
            )
            context = kwargs.pop("context", "") or ""
            return_route = bool(kwargs.pop("return_route", False))

            transcribe_kwargs = {}
            if context:
                transcribe_kwargs["context"] = context

            start_time = time.time()
            with torch.no_grad():
                result = asr.infer(
                    audio_path,
                    language=language_for_call,
                    return_objects=True,
                    return_route=return_route,
                    **transcribe_kwargs,
                )

            # ``MegaASR.infer`` returns either a list[ASRTranscription] (when
            # ``return_objects=True`` and ``return_route=False``) or a dict
            # ``{"text": ..., "use_lora": ..., ...}`` (when ``return_route``
            # is True; in that case ``text`` is itself a list).
            route_info = None
            if isinstance(result, dict):
                route_info = {
                    "use_lora": result.get("use_lora"),
                    "degraded_prob": result.get("degraded_prob"),
                    "route_source": result.get("route_source"),
                }
                inner = result.get("text")
            else:
                inner = result

            if isinstance(inner, list) and inner:
                first = inner[0]
                raw_text = getattr(first, "text", None)
                pred_lang = getattr(first, "language", "")
                if raw_text is None:
                    raw_text = str(first)
                    pred_lang = ""
            elif inner is None:
                raw_text = ""
                pred_lang = ""
            else:
                raw_text = str(inner)
                pred_lang = ""

            text = _clean_text(raw_text)
            elapsed = time.time() - start_time
            logger.info(
                "ASR done in %.2fs, len=%d, lang=%r, route=%r, audio=%s",
                elapsed, len(text), pred_lang, route_info,
                os.path.basename(audio_path),
            )

            response_obj = {
                "content": text,
                "raw_text": raw_text,
                "language": pred_lang,
            }
            if route_info is not None:
                response_obj["route"] = route_info

            response = json.dumps(response_obj, ensure_ascii=False)

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{response}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 30)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                if retry:
                    logger.debug(
                        "close signal not received within 30s, will emit again"
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    main()
