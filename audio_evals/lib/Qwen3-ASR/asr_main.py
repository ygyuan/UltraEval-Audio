"""Subprocess entry point for Qwen3-ASR.

This file is launched by ``audio_evals.models.asr.qwen3_asr.Qwen3ASR`` (via
the ``@isolated`` decorator) inside an isolated virtualenv that has the
heavy Qwen3-ASR dependencies installed (transformers==4.57.6, torch==2.6.0,
qwen-omni-utils, ...).  It speaks the same line-based JSON protocol as the
other ASR wrappers (e.g. MiMo-V2.5-ASR / VibeVoice-ASR):

    request   :  ``<uuid>-> {"audio": "/path/to.wav", "kwargs": {...}}\n``
    response  :  ``<uuid>-> {"content": "...", "raw_text": "..."}\n``
    close     :  ``<uuid>-> close\n``

The actual model code lives in the official Qwen3-ASR repo
(``third_party/Qwen3-ASR``) and is installed as an editable package via
``-e ./third_party/Qwen3-ASR`` (see asr_requirements.txt).  We invoke
``Qwen3ASRModel.from_pretrained(...)`` and ``transcribe(audio=path,
language=...)`` to obtain a transcription.

Reference:
    https://github.com/Qwen/Qwen3-ASR
    third_party/Qwen3-ASR/examples/example_qwen3_asr_transformers.py
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

import torch  # noqa: E402


logger = logging.getLogger("qwen3-asr")
logging.basicConfig(
    level=os.environ.get("QWEN3_ASR_LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# Qwen3-ASR uses canonical capitalised language names (see
# third_party/Qwen3-ASR/qwen_asr/inference/utils.py::SUPPORTED_LANGUAGES).
# We accept the short codes used elsewhere in UltraEval-Audio and map them
# to the canonical form expected by ``Qwen3ASRModel.transcribe``.
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
    "ms": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "filipino": "Filipino",
    "fa": "Persian",
    "persian": "Persian",
    "el": "Greek",
    "greek": "Greek",
    "ro": "Romanian",
    "romanian": "Romanian",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "macedonian": "Macedonian",
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
    # Already canonical (or any other supported language not in alias map).
    return s[:1].upper() + s[1:].lower()


# Strip leading chat-template role markers (rarely appear after parse_asr_output,
# but we keep this safety net consistent with the other ASR wrappers).
_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)


def _clean_text(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)
    return _LEADING_ROLE_RE.sub("", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path or HF id of the Qwen3-ASR checkpoint",
    )
    parser.add_argument(
        "--forced_aligner", type=str, default="",
        help=(
            "Optional path/HF id of the Qwen3-ForcedAligner checkpoint. "
            "When provided, time stamps could be requested per-call via "
            "kwargs.return_time_stamps=True (defaults to False for plain "
            "ASR scoring)."
        ),
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Model dtype",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
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
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate per chunk",
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

    default_language = _normalize_language(args.language)
    logger.info(
        "Loading Qwen3-ASR: model=%s, dtype=%s, device=%s, attn=%s, "
        "default_language=%r",
        args.path, args.dtype, args.device, args.attn_implementation,
        default_language,
    )

    # Resolve dtype.
    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device == "cpu":
        dtype = torch.float32

    # Pick a sensible attention implementation.
    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        if args.device.startswith("cuda") and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"
        else:
            attn_impl = "sdpa"
    logger.info("Resolved attn_implementation=%s", attn_impl)

    # Lazy import so any import error is surfaced on stderr only after the
    # subprocess has been launched (matches MiMo-V2.5-ASR / VibeVoice-ASR).
    from qwen_asr import Qwen3ASRModel  # type: ignore

    # Resolve the local model path: prefer absolute path of an existing
    # local directory so HuggingFace cache lookup never goes online.
    model_path = args.path
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)

    forced_aligner_path = args.forced_aligner.strip() or None
    forced_aligner_kwargs = None
    if forced_aligner_path:
        if os.path.isdir(forced_aligner_path):
            forced_aligner_path = os.path.abspath(forced_aligner_path)
        forced_aligner_kwargs = dict(
            dtype=dtype,
            device_map=args.device,
        )
        # Only request flash_attention_2 for the aligner if the main model
        # also uses it — otherwise stick with sdpa for safety.
        if attn_impl == "flash_attention_2":
            forced_aligner_kwargs["attn_implementation"] = "flash_attention_2"

    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=args.device,
        attn_implementation=attn_impl,
        forced_aligner=forced_aligner_path,
        forced_aligner_kwargs=forced_aligner_kwargs,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"Model loaded from checkpoint: {model_path}", flush=True)
    logger.info("Qwen3-ASR model loaded successfully")

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
            # Allow per-request overriding of language / context / time stamps.
            req_language = kwargs.pop("language", None)
            language_for_call = (
                _normalize_language(req_language)
                if req_language is not None
                else default_language
            )
            context = kwargs.pop("context", "") or ""
            return_time_stamps = bool(kwargs.pop("return_time_stamps", False))
            if return_time_stamps and forced_aligner_path is None:
                logger.warning(
                    "return_time_stamps=True but no --forced_aligner provided; "
                    "falling back to plain ASR (no timestamps)."
                )
                return_time_stamps = False

            start_time = time.time()
            with torch.no_grad():
                results = asr.transcribe(
                    audio=audio_path,
                    context=context,
                    language=language_for_call,
                    return_time_stamps=return_time_stamps,
                )

            # ``transcribe`` always returns List[ASRTranscription] of length 1
            # for a single audio input.
            if not results:
                raw_text = ""
                pred_lang = ""
            else:
                raw_text = results[0].text or ""
                pred_lang = results[0].language or ""

            text = _clean_text(raw_text)
            elapsed = time.time() - start_time
            logger.info(
                "ASR done in %.2fs, len=%d, lang=%r, audio=%s",
                elapsed, len(text), pred_lang, os.path.basename(audio_path),
            )

            result = json.dumps(
                {
                    "content": text,
                    "raw_text": raw_text,
                    "language": pred_lang,
                },
                ensure_ascii=False,
            )

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
                # Wait up to 30s for the parent's close ACK.
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
