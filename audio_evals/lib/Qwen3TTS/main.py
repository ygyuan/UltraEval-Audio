import argparse
import json
import logging
import os
import re
import select
import sys
import tempfile
import time

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Hallucination / repetition guard
# --------------------------------------------------------------------------
# LLM-based TTS (especially smaller variants such as Qwen3-TTS-0.6B-Base)
# occasionally fails to emit EOS at the right place and continues to
# hallucinate or repeat the input text, producing audio many times longer
# than the target sentence. We detect this by comparing the synthesized
# audio duration against an estimate based on input text length, and retry
# with progressively more conservative sampling parameters when triggered.

# Languages whose unit is character-level (CJK + Thai). Others are treated
# as space-separated word-level languages.
_CJK_LANGS = {"chinese", "japanese", "korean", "thai", "zh", "ja", "ko", "th"}

# Generous per-unit duration upper bounds (much larger than typical speech
# rate) so that only true hallucinations are flagged.
_SEC_PER_CHAR_CJK = 0.6     # ~100 char/min lower bound
_SEC_PER_WORD_OTHER = 0.7   # ~85 word/min lower bound (English etc.)
_MIN_DURATION_BUDGET = 8.0  # seconds, for very short sentences


def _estimate_max_duration(text: str, language: str) -> float:
    """Return a conservative upper bound on the expected audio duration."""
    if not text:
        return _MIN_DURATION_BUDGET
    lang = (language or "").strip().lower()
    if lang in _CJK_LANGS:
        # Count non-whitespace chars (covers CJK + mixed punctuation).
        n_units = len(re.sub(r"\s+", "", text))
        budget = n_units * _SEC_PER_CHAR_CJK
    else:
        # Word-level languages: count whitespace-separated tokens.
        tokens = [t for t in re.split(r"\s+", text) if t]
        n_units = max(len(tokens), 1)
        budget = n_units * _SEC_PER_WORD_OTHER
    return max(_MIN_DURATION_BUDGET, budget)


# Each retry tightens the sampling distribution to suppress runaway
# generation. Keys override the user-supplied generation kwargs.
_RETRY_OVERRIDES = [
    {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15,
     "subtalker_temperature": 0.7, "subtalker_top_p": 0.9},
    {"temperature": 0.5, "top_p": 0.85, "repetition_penalty": 1.2,
     "subtalker_temperature": 0.5, "subtalker_top_p": 0.85, "do_sample": False,
     "subtalker_dosample": False},
]


def _generate_with_guard(generate_fn, base_kwargs, text, language):
    """Call ``generate_fn(**kwargs)`` with a hallucination guard.

    Returns ``(wavs, sr, n_attempts)``. Falls back to the last attempt's
    output even if it still exceeds the budget, so the pipeline never
    aborts on a single bad sample.
    """
    max_dur = _estimate_max_duration(text, language)
    last_wavs, last_sr = None, None
    attempts = 1 + len(_RETRY_OVERRIDES)
    for attempt in range(attempts):
        kwargs = dict(base_kwargs)
        if attempt > 0:
            override = _RETRY_OVERRIDES[attempt - 1]
            # Only override keys the caller actually supports (i.e. already
            # present, or generation kwargs that the model accepts).
            for k, v in override.items():
                kwargs[k] = v
            logger.warning(
                "Hallucination guard: retry %d/%d with stricter params %s",
                attempt, attempts - 1, override,
            )
        wavs, sr = generate_fn(**kwargs)
        last_wavs, last_sr = wavs, sr
        dur = len(wavs[0]) / sr if sr > 0 else 0.0
        if dur <= max_dur:
            if attempt > 0:
                logger.info(
                    "Hallucination guard: retry succeeded at attempt %d "
                    "(dur=%.2fs, budget=%.2fs)",
                    attempt, dur, max_dur,
                )
            return wavs, sr, attempt + 1
        logger.warning(
            "Hallucination guard: audio dur=%.2fs exceeds budget=%.2fs "
            "(attempt %d/%d, text_len=%d)",
            dur, max_dur, attempt + 1, attempts, len(text or ""),
        )
    logger.error(
        "Hallucination guard: all %d attempts exceeded budget=%.2fs, "
        "returning last attempt as-is.",
        attempts, max_dur,
    )
    return last_wavs, last_sr, attempts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to Qwen3-TTS model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="custom_voice",
        choices=["custom_voice", "voice_design", "voice_clone"],
        help="Generation mode: custom_voice, voice_design, or voice_clone",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on",
    )
    args = parser.parse_args()

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading Qwen3-TTS model from {args.path} with dtype={args.dtype}")
    
    # Try to use flash attention if available
    try:
        model = Qwen3TTSModel.from_pretrained(
            args.path,
            device_map=args.device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        logger.info("Loaded with flash_attention_2")
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2: {e}, falling back to default")
        model = Qwen3TTSModel.from_pretrained(
            args.path,
            device_map=args.device,
            dtype=dtype,
        )
    
    logger.info(f"Qwen3-TTS model loaded successfully in {args.mode} mode")
    
    # Get supported speakers and languages for custom voice mode
    if args.mode == "custom_voice":
        try:
            speakers = model.get_supported_speakers()
            languages = model.get_supported_languages()
            logger.info(f"Supported speakers: {speakers}")
            logger.info(f"Supported languages: {languages}")
        except Exception as e:
            logger.warning(f"Could not get supported speakers/languages: {e}")

    # Enable RTF tracking from environment variable
    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

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
            x = json.loads(prompt[anchor + 2:])

            # Record start time for RTF calculation
            torch.cuda.synchronize()
            start_time = time.time()

            # Extract common parameters
            text = x.pop("text")
            language = x.pop("language", "Auto")
            
            if args.mode == "custom_voice":
                # Custom voice generation
                speaker = x.pop("speaker", "Vivian")
                instruct = x.pop("instruct", None)
                generate_kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": speaker,
                }
                generate_kwargs.update(x)
                if instruct:
                    generate_kwargs["instruct"] = instruct
                logger.info(f"generate_custom_voice kwargs: {generate_kwargs}")
                wavs, sr, n_attempts = _generate_with_guard(
                    model.generate_custom_voice, generate_kwargs, text, language,
                )

            elif args.mode == "voice_design":
                # Voice design generation
                instruct = x.pop("instruct", "")
                logger.info(f"voice_design: text: {text}, language: {language}, instruct: {instruct}, **x: {x}")
                generate_kwargs = {
                    "text": text,
                    "language": language,
                    "instruct": instruct,
                }
                generate_kwargs.update(x)
                wavs, sr, n_attempts = _generate_with_guard(
                    model.generate_voice_design, generate_kwargs, text, language,
                )

            elif args.mode == "voice_clone":
                # Voice clone generation
                ref_audio = x.pop("prompt_audio")
                ref_text = x.pop("prompt_text")

                if ref_audio is None:
                    raise ValueError("ref_audio is required for voice_clone mode")
                logger.info(f"ref_audio: {ref_audio}, ref_text: {ref_text}, text: {text}, language: {language}, **x: {x}")
                generate_kwargs = {
                    "text": text,
                    "language": language,
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                }
                generate_kwargs.update(x)
                wavs, sr, n_attempts = _generate_with_guard(
                    model.generate_voice_clone, generate_kwargs, text, language,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # Record end time
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time

            # Save output to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wavs[0], sr)
                output_path = f.name

            # Return result with optional RTF
            if enable_rtf == 1:
                audio_duration = len(wavs[0]) / sr
                rtf = inference_time / audio_duration if audio_duration > 0 else 0
                result = json.dumps({"audio": output_path, "RTF": rtf})
                logger.info(
                    f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                )
            else:
                result = output_path

            # Output result with retry mechanism
            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
