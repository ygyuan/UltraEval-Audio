"""
VibeVoice ASR main script for isolated subprocess execution.

This script loads the VibeVoice ASR model and handles speech-to-text
requests via stdin/stdout communication with the parent process.

Reference:
    https://github.com/microsoft/VibeVoice
    init_model/microsoft/VibeVoice/demo/vibevoice_asr_inference_from_file.py
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
# typically has no network access, and we provide all model / tokenizer
# files locally.  These env vars must be set BEFORE importing transformers /
# huggingface_hub so that they take effect during import-time HTTP calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Regex to strip bracketed ASCII structural/non-speech tags such as
# [Music], [Lyric], [Unintelligible Speech], [Vocal], [Noise], etc.
# These tags are produced by VibeVoice-ASR for non-speech segments and
# inflate CER when compared against plain-text references.
_BRACKET_TAG_RE = re.compile(r"\[\s*[A-Za-z][A-Za-z0-9 _\-/]*\s*\]")
_WS_RE = re.compile(r"\s+")

# Regex to recover Content fields from a raw model output that *looks like*
# JSON list of segments but failed structured parsing, e.g.:
#   assistant\n[{"Start":0,"End":1.0,"Speaker":0,"Content":"hello"}]
_CONTENT_FIELD_RE = re.compile(
    r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"'
)
# Same as above but tolerates a *truncated* trailing Content field
# (no closing quote) which happens when the model hits ``max_new_tokens``
# in the middle of a long repetition collapse.
#
# We make the trailing match *bounded* so it stops at the next plausible
# JSON structural boundary (e.g. ``,"Start"`` of the next segment, or the
# closing ``}]`` of the segment list).  Without these stoppers, the regex
# would happily swallow trailing JSON noise into the recovered Content,
# polluting the prediction.  Order of stoppers in the alternation does not
# matter -- whichever appears first in the string wins via the lazy ``*?``
# on the captured group.
_CONTENT_FIELD_OPEN_RE = re.compile(
    r'"Content"\s*:\s*"'
    r'((?:[^"\\]|\\.)*?)'
    r'(?:,\s*"Start"|\}\s*[,\]]|$)'
)
# Strip leading chat-template role tokens (e.g. "assistant\n", "<|assistant|>")
# that some HF chat templates leave at the very beginning of decoded text.
_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)

# Default context_info strings per language to guide the model.
_LANG_CONTEXT = {
    "zh": (
        "Please transcribe in Chinese (Mandarin) only. "
        "Do not output lyrics, foreign languages, or non-speech tags."
    ),
    "en": (
        "Please transcribe in English only. "
        "Do not output lyrics or non-speech tags."
    ),
}

# Used when no explicit ``--language`` flag is provided: still instruct the
# model to transcribe in the audio's own language (avoiding spurious foreign
# output observed on the WenetSpeech / KeSpeech tests when language was unset).
_DEFAULT_CONTEXT = (
    "Please transcribe the audio in its original spoken language. "
    "Do not translate, do not output lyrics, and do not output non-speech tags."
)


def _strip_bracketed_tags(text: str) -> str:
    """Remove [Xxx] / [Xxx Yyy] ASCII structural tags and collapse whitespace."""
    cleaned = _BRACKET_TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", cleaned).strip()


def _segments_to_plain_text(segments):
    """Concatenate the ``text`` field of each transcription segment.

    Bracketed structural tags (e.g. ``[Lyric]``, ``[Unintelligible Speech]``,
    ``[Music]``) are stripped from each segment's text before concatenation.
    Segments whose text is entirely a structural tag (i.e. empty after
    stripping) are skipped.
    """
    if not segments:
        return ""
    parts = []
    for seg in segments:
        if isinstance(seg, dict):
            text = seg.get("text") or seg.get("Content") or ""
            if text:
                text = _strip_bracketed_tags(str(text))
                if text:
                    parts.append(text)
    return " ".join(parts).strip()


def _clean_asr_text(text: str) -> str:
    """Remove bracketed structural tags from a VibeVoice-ASR prediction."""
    if not text:
        return ""
    cleaned = _BRACKET_TAG_RE.sub(" ", text)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """Collapse pathological repeated substrings produced by the LM.

    VibeVoice-ASR occasionally falls into a *repetition collapse*: it keeps
    emitting the same phrase (e.g. ``"好了"``, ``"我那个朋友呢"``,
    ``"Hallelujah,"``) until ``max_new_tokens`` is exhausted. The resulting
    32k+ char string then dominates CER even after JSON parsing, because
    the underlying ``"Content"`` field really did contain that loop.

    This function detects any consecutive repetition of a short n-gram
    (1-20 chars) and keeps at most ``max_repeat`` copies of it. It is a
    safety net only; legitimate text never repeats the same 5-gram more
    than 2-3 times in a row, while pathological output repeats it hundreds
    of times.

    For *very* long inputs (e.g. 5k+ chars, almost certainly pathological)
    we additionally collapse any 2+ char unit repeated 6+ times down to a
    single copy, since at that scale even ``max_repeat`` copies are still
    enough to wreck the CER.
    """
    if not text or len(text) < 32:
        return text
    out = text
    # Iterate from longer units to shorter ones so longer repeats are caught
    # first.  ``(.{n})\1{k,}`` shrinks ``unit * (k+1+)`` to ``unit * max_repeat``.
    for unit_len in range(20, 0, -1):
        pattern = re.compile(
            r"(.{" + str(unit_len) + r"})(?:\1){" + str(max_repeat) + r",}",
            flags=re.DOTALL,
        )
        out = pattern.sub(lambda m: m.group(1) * max_repeat, out)

    # Aggressive pass for clearly-pathological remainders: any 2-20 char
    # unit still repeated 6+ times is collapsed to a single copy.  This
    # only kicks in if the output is suspiciously long (>2000 chars), so
    # legitimate transcriptions are never touched.
    if len(out) > 2000:
        for unit_len in range(20, 1, -1):
            pattern = re.compile(
                r"(.{" + str(unit_len) + r"})(?:\1){5,}",
                flags=re.DOTALL,
            )
            out = pattern.sub(lambda m: m.group(1), out)
    return out


def _get_audio_duration(audio_path: str):
    """Return the duration of ``audio_path`` in seconds, or ``None``.

    Tries ``soundfile`` first (handles wav/flac/ogg), then the stdlib
    ``wave`` module, then ``audioread`` as a last resort.  Failures are
    silent: callers must treat ``None`` as "unknown duration" and skip
    the duration-based truncation.
    """
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(audio_path)
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import wave

        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0 and frames > 0:
                return float(frames) / float(rate)
    except Exception:
        pass
    try:
        import audioread  # type: ignore

        with audioread.audio_open(audio_path) as af:
            return float(af.duration)
    except Exception:
        pass
    return None


# Rough upper bound on chars-per-second of natural speech, used to detect
# pathological output that vastly exceeds what the audio could contain.
# Mandarin: ~5-6 chars/s normal, ~10 chars/s fast; English: ~15-20 chars/s
# normal, ~25 chars/s fast.  We use the larger English bound as the default
# so we never falsely flag legitimate speech.
_MAX_CHARS_PER_SEC = 25.0
# Absolute floor for very short clips: even a 1s clip should never be
# allowed to produce more than ~200 chars of transcription.
_MIN_LEN_CAP = 200


def _truncate_by_audio_duration(text: str, audio_path: str) -> str:
    """Hard upper-bound the transcription length using audio duration.

    If ``text`` exceeds ``duration_sec * _MAX_CHARS_PER_SEC * 2`` (a 2x
    safety margin), it is almost certainly a repetition-collapse output
    that survived earlier guards.  We first try to compress repeats one
    more time; if that is still not enough, we hard-truncate the string
    to the duration-derived cap.  This guarantees that a single bad
    sample can not contribute thousands-of-percent CER to the dataset
    average even if every other guard fails.
    """
    if not text:
        return text
    duration = _get_audio_duration(audio_path)
    if duration is None or duration <= 0:
        return text
    cap = max(_MIN_LEN_CAP, int(duration * _MAX_CHARS_PER_SEC * 2))
    if len(text) <= cap:
        return text
    logger.warning(
        "ASR output (len=%d) exceeds audio-duration cap (%.2fs -> cap=%d); "
        "compressing repeats and truncating.",
        len(text),
        duration,
        cap,
    )
    # Try one more aggressive compression pass before hard truncation.
    compressed = _collapse_repeats(text, max_repeat=2)
    if len(compressed) <= cap:
        return compressed
    return compressed[:cap]


def _select_attn_implementation(device: str, requested: str) -> str:
    """Pick a sensible attention implementation given the device."""
    if requested != "auto":
        return requested
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the VibeVoice-ASR model checkpoint",
    )
    parser.add_argument(
        "--language_model_pretrained_name",
        type=str,
        default="init_model/Qwen2.5-7B-Instruct",
        help=(
            "Tokenizer / language model name passed to VibeVoiceASRProcessor."
            " Use a local path when the host has no internet access."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=list(_LANG_CONTEXT.keys()) + [None],
        help=(
            "Target transcription language (e.g. 'zh', 'en'). "
            "When set, the corresponding context_info hint is passed to the "
            "processor to constrain the output language."
        ),
    )
    args = parser.parse_args()

    # Resolve the context_info hint for the target language.  When no language
    # is provided, fall back to a generic "transcribe in original language"
    # hint so the model does not freely switch into other languages
    # (observed: Mandarin audio being transcribed as English / Japanese /
    # Russian when context_info=None).
    if args.language:
        context_info = _LANG_CONTEXT.get(args.language)
        logger.info(
            "Using context_info for language=%s: %s", args.language, context_info
        )
    else:
        context_info = _DEFAULT_CONTEXT
        logger.info(
            "No --language provided; using default context_info: %s", context_info
        )

    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device == "cpu":
        dtype = torch.float32

    attn_impl = _select_attn_implementation(args.device, args.attn_implementation)
    logger.info(
        "Loading VibeVoice-ASR model from %s (dtype=%s, attn=%s, device=%s)",
        args.path,
        args.dtype,
        attn_impl,
        args.device,
    )

    # Resolve language_model_pretrained_name: prefer absolute path of an
    # existing local directory so HuggingFace cache lookup never goes online.
    lm_name = args.language_model_pretrained_name
    if os.path.isdir(lm_name):
        lm_name = os.path.abspath(lm_name)
    logger.info("Using language_model_pretrained_name=%s", lm_name)

    processor = VibeVoiceASRProcessor.from_pretrained(
        args.path,
        language_model_pretrained_name=lm_name,
        local_files_only=True,
    )

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.path,
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(args.device)
    model.eval()
    logger.info("VibeVoice-ASR model loaded successfully on %s", args.device)

    while True:
        try:
            prompt = input()
        except EOFError:
            break

        try:
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contain '->' but got {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            payload = json.loads(prompt[anchor + 2:])

            audio_path = payload.get("audio")
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
            max_new_tokens = int(kwargs.pop("max_new_tokens", args.max_new_tokens))
            temperature = float(kwargs.pop("temperature", 0.0))
            top_p = float(kwargs.pop("top_p", 1.0))
            num_beams = int(kwargs.pop("num_beams", 1))
            do_sample = bool(kwargs.pop("do_sample", temperature > 0))

            # Anti-repetition guards: VibeVoice-ASR is prone to falling into
            # infinite-repetition loops on hard / noisy / non-speech audio
            # (e.g. emitting "好了好了好了..." until ``max_new_tokens``).
            # ``repetition_penalty`` discourages repeating any token, and
            # ``no_repeat_ngram_size`` outright bans repeating long n-grams.
            # Both are overridable via per-request kwargs.
            repetition_penalty = float(
                kwargs.pop("repetition_penalty", 1.1)
            )
            no_repeat_ngram_size = int(
                kwargs.pop("no_repeat_ngram_size", 10)
            )

            start_time = time.time()
            inputs = processor(
                audio=[audio_path],
                sampling_rate=None,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
                context_info=context_info,
            )

            inputs = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": processor.pad_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if repetition_penalty and repetition_penalty != 1.0:
                generation_config["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                generation_config["no_repeat_ngram_size"] = no_repeat_ngram_size
            if num_beams > 1:
                generation_config["num_beams"] = num_beams
                generation_config["do_sample"] = False
            else:
                generation_config["do_sample"] = do_sample
                if do_sample:
                    generation_config["temperature"] = temperature
                    generation_config["top_p"] = top_p

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_config)

            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_length:]
            eos_positions = (
                generated_ids == processor.tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[: eos_positions[0] + 1]

            raw_text = processor.decode(generated_ids, skip_special_tokens=True)

            try:
                segments = processor.post_process_transcription(raw_text)
            except Exception as e:
                logger.warning("post_process_transcription failed: %s", e)
                segments = []

            plain_text = _segments_to_plain_text(segments)
            if not plain_text:
                # Structured parsing failed.  Try a best-effort recovery
                # by extracting every "Content":"..." field from the raw
                # text (it usually looks like:
                #   assistant\n[{"Start":0,"End":1.0,"Content":"hello"}]
                # ).  This avoids returning the JSON-ish blob verbatim,
                # which previously produced 100% CER on every parse-fail
                # sample.
                fallback = raw_text or ""
                # Drop leading chat-template role markers (e.g. "assistant\n").
                fallback = _LEADING_ROLE_RE.sub("", fallback)
                content_matches = list(_CONTENT_FIELD_RE.findall(fallback))
                # Also recover a *truncated* trailing Content field that
                # was cut off by ``max_new_tokens`` (no closing quote).
                # Without this, repetition-collapse outputs that hit the
                # token limit produce no closed Content match and the
                # entire 32k+ JSON-ish blob would be returned verbatim,
                # producing thousands-of-percent CER on a single sample.
                tail = _CONTENT_FIELD_OPEN_RE.search(fallback)
                if tail is not None:
                    content_matches.append(tail.group(1))
                if content_matches:
                    # Unescape JSON string escapes (\" \\ \n ...).
                    decoded_parts = []
                    for m in content_matches:
                        try:
                            decoded = json.loads(f'"{m}"')
                        except Exception:
                            decoded = m
                        # Per-segment repeat compression: a single Content
                        # field can itself contain thousands of repeated
                        # phrases ("好了好了好了..."), and concatenating
                        # them before compression makes the longer-unit
                        # detection of ``_collapse_repeats`` slower and
                        # less accurate.  Compress here first.
                        decoded = _collapse_repeats(decoded)
                        decoded_parts.append(decoded)
                    plain_text = " ".join(
                        p.strip() for p in decoded_parts if p and p.strip()
                    )
                if not plain_text:
                    # Last resort: never hand back a verbatim JSON-ish blob
                    # (always starts with ``[{`` or ``{"`` and is dominated
                    # by structural tokens).  Returning an empty string
                    # yields 100% CER for that single sample, which is
                    # still much better than the previous behaviour where
                    # one bad sample contributed thousands-of-percent CER
                    # and skewed the dataset average.
                    stripped = fallback.strip()
                    if stripped.startswith("[") or stripped.startswith("{"):
                        logger.warning(
                            "Discarding unparseable JSON-ish ASR output "
                            "(len=%d) to avoid CER explosion",
                            len(stripped),
                        )
                        plain_text = ""
                    else:
                        plain_text = stripped

            # Strip "[Music]" / "[Lyric]" / "[Vocal]" / ... structural tags
            # that VibeVoice-ASR may emit but are never present in the
            # reference transcriptions of normal ASR benchmarks.
            plain_text = _clean_asr_text(plain_text)
            # Final safety net: collapse pathological repetitions that
            # survived the language-model's anti-repetition guards.
            plain_text = _collapse_repeats(plain_text)
            # Hard upper bound based on the source audio's duration: if
            # the prediction is still impossibly long (e.g. a 3s clip
            # produced 8000+ chars), compress aggressively and truncate.
            plain_text = _truncate_by_audio_duration(plain_text, audio_path)

            elapsed = time.time() - start_time
            logger.info(
                "ASR done in %.2fs, %d segments, text length=%d",
                elapsed,
                len(segments),
                len(plain_text),
            )

            # Wrap into JSON for the parent process.
            result = json.dumps(
                {
                    "content": plain_text,
                    "raw_text": raw_text,
                    "segments": segments,
                },
                ensure_ascii=False,
            )

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
                # Wait up to 30s for the parent's close ACK.  The previous
                # 5s budget was too short under high concurrency (8 workers
                # x N GPUs) and produced a flood of false "not found close
                # signal" warnings on stderr.
                rlist, _, _ = select.select([sys.stdin], [], [], 30)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                if retry:
                    # Demote to a debug-level log on stderr so it does not
                    # show up as a process error in the parent's stderr
                    # routing logic.
                    logger.debug(
                        "close signal not received within 30s, will emit again"
                    )
        except Exception as e:
            import traceback

            traceback.print_exc()
            err_str = str(e)
            print(f"Error: {err_str}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            # Some errors corrupt the CUDA context (e.g. "CUDA error:
            # unspecified launch failure", "device-side assert triggered",
            # "out of memory") and *every* subsequent kernel launch in this
            # process will keep failing with the same error.  In that
            # situation the worker has to die so the parent's
            # ``ensure_process_alive`` can spawn a fresh subprocess with a
            # clean CUDA context.  Otherwise the worker turns into a
            # "zombie" that returns the same error for thousands of
            # downstream samples (observed: 76% fail rate on
            # vibevoice-asr-zh / asr_lianghui).
            fatal_keywords = (
                "CUDA error",
                "CUDA out of memory",
                "out of memory",
                "device-side assert",
                "an illegal memory access",
                "CUBLAS_STATUS",
                "CUDNN_STATUS",
                "NCCL",
                "no kernel image is available",
                "Driver error",
            )
            if any(kw in err_str for kw in fatal_keywords):
                logger.error(
                    "Fatal CUDA/runtime error detected, exiting subprocess "
                    "so parent can restart with a clean context: %s",
                    err_str,
                )
                # Best-effort GPU cleanup; ignore any further failures
                # (CUDA context is already broken at this point).
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                os._exit(1)


if __name__ == "__main__":
    main()
