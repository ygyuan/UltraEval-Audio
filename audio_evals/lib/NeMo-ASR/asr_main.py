"""Subprocess entry point for NVIDIA NeMo ASR models.

This file is launched by ``audio_evals.models.asr.nemo_asr.NemoASR`` (via
the ``@isolated`` decorator) inside an isolated virtualenv that has the
heavy NeMo dependencies installed (nemo_toolkit[asr], torch, ...).
It speaks the same line-based JSON protocol as the other ASR wrappers
(e.g. Qwen3-ASR / MiMo-V2.5-ASR / VibeVoice-ASR):

    request   :  ``<uuid>-> {"audio": "/path/to.wav", "kwargs": {...}}\n``
    response  :  ``<uuid>-> {"content": "...", "raw_text": "..."}\n``
    close     :  ``<uuid>-> close\n``

Supported NeMo ASR architectures (auto-detected from the checkpoint):
    * ``EncDecRNNTBPEModel``       — parakeet-tdt-0.6b-v3 (CTC/RNNT/TDT)
    * ``EncDecMultiTaskModel``     — canary-1b-v2 (multi-task: ASR + AST)

Reference:
    https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
    https://huggingface.co/nvidia/canary-1b-v2
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Reduce NeMo's verbose logging on stderr.
os.environ.setdefault("NEMO_LOGGING_LEVEL", "WARNING")

import torch  # noqa: E402


logger = logging.getLogger("nemo-asr")
logging.basicConfig(
    level=os.environ.get("NEMO_ASR_LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# Canary / parakeet-tdt-0.6b-v3 use ISO 639-1 short codes (e.g. ``en``,
# ``zh``, ``de``).  We accept a few common aliases and pass them through
# unchanged otherwise.
_LANG_ALIAS = {
    "chinese": "zh",
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "dutch": "nl",
    "polish": "pl",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "czech": "cs",
    "greek": "el",
    "hungarian": "hu",
    "romanian": "ro",
    "turkish": "tr",
    "ukrainian": "uk",
    "auto": None,
    "": None,
}


def _normalize_language(language):
    """Map a user-provided ``language`` to the canonical short code.

    Returns ``None`` for empty / "auto" input (auto-detect / model default).
    """
    if language is None:
        return None
    s = str(language).strip()
    if not s:
        return None
    key = s.lower()
    if key in _LANG_ALIAS:
        return _LANG_ALIAS[key]
    # Already canonical short code.
    return key


_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)


def _clean_text(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)
    return _LEADING_ROLE_RE.sub("", text).strip()


def _detect_model_class(path):
    """Best-effort guess at the NeMo model class from the path.

    Returns one of ``"multitask"`` / ``"rnnt"`` / ``"ctc"`` / ``"unknown"``.
    """
    name = os.path.basename(path.rstrip("/")).lower()
    if "canary" in name:
        return "multitask"
    if "parakeet" in name:
        # parakeet-tdt / parakeet-rnnt all use EncDecRNNTBPEModel.
        # parakeet-ctc would use EncDecCTCModelBPE; we treat both
        # rnnt / tdt as RNNT here (which is correct for v3).
        if "ctc" in name and "tdt" not in name and "rnnt" not in name:
            return "ctc"
        return "rnnt"
    return "unknown"


def _load_nemo_model(path, model_class, device, dtype):
    """Load a NeMo ASR model from a local .nemo file or a directory.

    Args:
        path: directory containing a single .nemo file, or path to a
            .nemo file directly, or a HuggingFace repo id.
        model_class: ``"multitask"`` / ``"rnnt"`` / ``"ctc"`` / ``"unknown"``.
        device: ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``.
        dtype: torch dtype (only applied to fp16/bf16).

    Returns:
        Loaded NeMo ASR model in eval mode on the requested device.
    """
    # Lazy-import nemo so failures are visible only on stderr after launch.
    from nemo.collections.asr.models import ASRModel  # type: ignore

    # Resolve the actual checkpoint file/dir.
    resolved = path
    if os.path.isdir(path):
        # Prefer a .nemo file if present; otherwise let NeMo handle the dir.
        nemo_files = [f for f in os.listdir(path) if f.endswith(".nemo")]
        if len(nemo_files) == 1:
            resolved = os.path.join(path, nemo_files[0])
        elif len(nemo_files) > 1:
            # Pick the largest one (typically the main checkpoint).
            nemo_files.sort(
                key=lambda f: os.path.getsize(os.path.join(path, f)),
                reverse=True,
            )
            resolved = os.path.join(path, nemo_files[0])
            logger.warning(
                "Multiple .nemo files found in %s, using %s",
                path, nemo_files[0],
            )

    logger.info("Loading NeMo ASR model from %s (class hint=%s)",
                resolved, model_class)

    # Use the generic ASRModel loader — it inspects the checkpoint and
    # constructs the correct subclass automatically.
    if resolved.endswith(".nemo") and os.path.isfile(resolved):
        model = ASRModel.restore_from(
            restore_path=resolved,
            map_location=torch.device(device),
        )
    else:
        # Fallback: HuggingFace repo id or remote name.
        model = ASRModel.from_pretrained(
            model_name=resolved,
            map_location=torch.device(device),
        )

    model = model.to(device)
    model.eval()

    # Best-effort cast for inference; some NeMo modules don't fully
    # support bf16 — fall back to fp32 on errors.
    if dtype in (torch.float16, torch.bfloat16):
        try:
            model = model.to(dtype)
        except Exception as e:
            logger.warning(
                "Failed to cast NeMo model to %s, keeping fp32: %s", dtype, e,
            )

    return model


def _transcribe_one(model, audio_path, model_class, language, max_new_tokens):
    """Run inference on a single audio file and return the transcription.

    Returns a tuple ``(content, raw_text, language)``.
    """
    # Build transcribe kwargs based on the architecture.
    transcribe_kwargs = {
        "audio": [audio_path],
        "batch_size": 1,
        "return_hypotheses": False,
    }
    pred_lang = language or ""

    # Multi-task (canary): needs source_lang / target_lang / task fields.
    if model_class == "multitask":
        # When no language is provided, default to English ASR (canary-1b-v2
        # primary use-case).  Translation can be requested via kwargs.task.
        lang = language or "en"
        transcribe_kwargs["source_lang"] = lang
        transcribe_kwargs["target_lang"] = lang
        transcribe_kwargs["task"] = "asr"
        transcribe_kwargs["pnc"] = "yes"
        pred_lang = lang

    with torch.no_grad():
        results = model.transcribe(**transcribe_kwargs)

    # ``transcribe`` returns either ``List[str]`` (CTC / RNNT) or a list
    # of ``Hypothesis`` objects when ``return_hypotheses=True``.  For
    # canary it returns ``List[str]`` of transcripts.
    if not results:
        raw_text = ""
    else:
        first = results[0]
        # Some NeMo versions return [List[str], List[str]] for RNNT+TDT
        # (greedy + nbest); in that case unwrap.
        if isinstance(first, list) and first:
            first = first[0]
        if hasattr(first, "text"):
            raw_text = first.text or ""
        else:
            raw_text = str(first) if first is not None else ""

    return _clean_text(raw_text), raw_text, pred_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="Local path to the NeMo ASR model directory or .nemo file.",
    )
    parser.add_argument(
        "--model_class", type=str, default="auto",
        choices=["auto", "multitask", "rnnt", "ctc"],
        help=(
            "Architecture hint. 'auto' tries to infer from the path name. "
            "Use 'multitask' for canary-1b-v2 and 'rnnt' for "
            "parakeet-tdt-0.6b-v3."
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
        "--language", type=str, default="",
        help=(
            "Optional default target language (e.g. 'en', 'zh', 'de'). "
            "Empty string means auto-detect / model default."
        ),
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate (used by multi-task models).",
    )
    args = parser.parse_args()

    # Resolve dtype.
    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device == "cpu":
        dtype = torch.float32

    default_language = _normalize_language(args.language)

    # Resolve architecture.
    model_class = args.model_class
    if model_class == "auto":
        model_class = _detect_model_class(args.path)
        if model_class == "unknown":
            logger.warning(
                "Could not auto-detect NeMo model class from path %s, "
                "defaulting to 'rnnt'.", args.path,
            )
            model_class = "rnnt"

    logger.info(
        "Loading NeMo-ASR: model=%s, dtype=%s, device=%s, class=%s, "
        "default_language=%r",
        args.path, args.dtype, args.device, model_class, default_language,
    )

    # Resolve the local model path.
    model_path = args.path
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        model_path = os.path.abspath(model_path)

    asr = _load_nemo_model(model_path, model_class, args.device, dtype)
    print(f"Model loaded from checkpoint: {model_path}", flush=True)
    logger.info("NeMo-ASR model loaded successfully (class=%s)", model_class)

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

            start_time = time.time()
            text, raw_text, pred_lang = _transcribe_one(
                asr, audio_path, model_class, language_for_call,
                args.max_new_tokens,
            )
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
