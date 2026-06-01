"""Subprocess entry point for MiMo-V2.5-ASR.

This file is launched by ``audio_evals.models.asr.mimo_asr.MiMoASR`` (via the
``@isolated`` decorator) inside an isolated virtualenv that has the heavy
MiMo-V2.5-ASR dependencies installed.  It speaks the same line-based
JSON protocol as the other ASR wrappers (e.g. GLM-ASR / VibeVoice-ASR):

    request   :  ``<uuid>-> {"audio": "/path/to.wav", "kwargs": {...}}\n``
    response  :  ``<uuid>-> {"content": "...", "raw_text": "..."}\n``
    close     :  ``<uuid>-> close\n``

The actual model code lives in the official MiMo-V2.5-ASR repo
(``third_party/MiMo-V2.5-ASR``) and exposes ``MimoAudio`` with an
``asr_sft(audio, audio_tag="")`` method, where ``audio_tag`` is one of
``""`` / ``"<chinese>"`` / ``"<english>"``.
"""

import argparse
import json
import logging
import os
import re
import select
import sys
import time


logger = logging.getLogger("mimo-v25-asr")
logging.basicConfig(
    level=os.environ.get("MIMO_ASR_LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Path setup: make ``from src.mimo_audio.mimo_audio import MimoAudio`` work
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
# UltraEval-Audio project root (.../UltraEval-Audio/).
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_FILE, "..", "..", "..", ".."))
_DEFAULT_REPO_ROOT = os.path.join(
    _PROJECT_ROOT, "third_party", "MiMo-V2.5-ASR"
)
_REPO_ROOT = os.environ.get("MIMO_V25_ASR_REPO", _DEFAULT_REPO_ROOT)
_REPO_ROOT = os.path.abspath(_REPO_ROOT)
if not os.path.isdir(_REPO_ROOT):
    raise RuntimeError(
        f"MiMo-V2.5-ASR repository not found at {_REPO_ROOT}. "
        "Set env MIMO_V25_ASR_REPO to its real location."
    )
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Default tokenizer: shared with the MiMo-Audio family.
_DEFAULT_TOKENIZER = os.path.join(
    _PROJECT_ROOT, "init_model", "XiaomiMiMo", "MiMo-Audio-Tokenizer"
)


# ---------------------------------------------------------------------------
# Output post-processing: strip residual chat-template / role markers so the
# main framework receives a clean transcription ready for WER/CER scoring.
# ---------------------------------------------------------------------------
_LEADING_ROLE_RE = re.compile(
    r"^(?:<\|im_start\|>)?\s*assistant\s*\n", flags=re.IGNORECASE
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_SPECIAL_TOKENS = (
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
    "<|empty|>", "<|eot|>", "<|eostm|>",
    "<chinese>", "<english>",
)


def _clean_text(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _LEADING_ROLE_RE.sub("", text)
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def _normalize_audio_tag(language):
    """Map a high level ``language`` arg to MiMo-V2.5-ASR's ``audio_tag``."""
    if not language:
        return ""
    lang = str(language).strip().lower()
    if lang in ("zh", "zh-cn", "chinese", "<chinese>", "cn"):
        return "<chinese>"
    if lang in ("en", "en-us", "english", "<english>"):
        return "<english>"
    if lang in ("auto", ""):
        return ""
    # Unknown languages: forward as-is (most likely empty / auto).
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path or HF id of the MiMo-V2.5-ASR checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=_DEFAULT_TOKENIZER,
        help="Path or HF id of the MiMo-Audio-Tokenizer checkpoint",
    )
    parser.add_argument(
        "--language", type=str, default="",
        help="Optional language hint: 'zh' / 'en' / '' (auto)",
    )
    args = parser.parse_args()

    audio_tag = _normalize_audio_tag(args.language)
    logger.info(
        "Loading MiMo-V2.5-ASR: model=%s, tokenizer=%s, audio_tag=%r",
        args.path, args.tokenizer_path, audio_tag,
    )

    # Import lazily so any import error is reported on stderr only after the
    # subprocess has been launched (matches GLM-ASR / VibeVoice-ASR behaviour).
    from src.mimo_audio.mimo_audio import MimoAudio  # type: ignore

    model = MimoAudio(args.path, args.tokenizer_path)
    print(f"Model loaded from checkpoint: {args.path}", flush=True)
    logger.info("MiMo-V2.5-ASR model loaded successfully")

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
            # Allow per-request overriding of the language tag.
            req_language = kwargs.pop("language", None)
            tag_for_this_call = (
                _normalize_audio_tag(req_language)
                if req_language is not None
                else audio_tag
            )

            start_time = time.time()
            raw_text = model.asr_sft(audio_path, audio_tag=tag_for_this_call)
            text = _clean_text(raw_text)
            elapsed = time.time() - start_time
            logger.info(
                "ASR done in %.2fs, len=%d, audio=%s",
                elapsed, len(text), os.path.basename(audio_path),
            )

            result = json.dumps(
                {"content": text, "raw_text": raw_text or ""},
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
