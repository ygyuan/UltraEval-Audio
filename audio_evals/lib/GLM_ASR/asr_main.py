"""
GLM-ASR main script for isolated subprocess execution.

This script loads the GLM-ASR (e.g. GLM-ASR-Nano-2512) model and handles
speech-to-text requests via stdin/stdout communication with the parent
process.

Reference:
    https://github.com/zai-org/GLM-ASR
    init_model/zai-org/GLM-ASR/inference.py
    init_model/zai-org/GLM-ASR-Nano-2512/README.md
"""

import argparse
import json
import logging
import os
import re
import select
import sys
import time

# Force fully offline mode for the isolated subprocess.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_BRACKET_TAG_RE = re.compile(r"\[\s*[A-Za-z][A-Za-z0-9 _\-/]*\s*\]")
_WS_RE = re.compile(r"\s+")
_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)

_LANG_PROMPT = {
    "zh": "Please transcribe this audio into text in Chinese (Mandarin).",
    "en": "Please transcribe this audio into text in English.",
}
_DEFAULT_PROMPT = "Please transcribe this audio into text"


def _clean_asr_text(text):
    if not text:
        return ""
    cleaned = _BRACKET_TAG_RE.sub(" ", text)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _collapse_repeats(text, max_repeat=3):
    if not text or len(text) < 32:
        return text
    out = text
    for unit_len in range(20, 0, -1):
        pattern = re.compile(
            r"(.{" + str(unit_len) + r"})(?:\1){" + str(max_repeat) + r",}",
            flags=re.DOTALL,
        )
        out = pattern.sub(lambda m: m.group(1) * max_repeat, out)
    if len(out) > 2000:
        for unit_len in range(20, 1, -1):
            pattern = re.compile(
                r"(.{" + str(unit_len) + r"})(?:\1){5,}",
                flags=re.DOTALL,
            )
            out = pattern.sub(lambda m: m.group(1), out)
    return out


def _select_attn_implementation(device, requested):
    if requested != "auto":
        return requested
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def _load_audio_array(audio_path, target_sr):
    """Decode an audio file into a mono ``float32`` numpy array at ``target_sr``.

    GLM-ASR's processor would internally use ``torchcodec`` (which links
    against system FFmpeg) to decode files passed as paths.  In many
    isolated environments ``libtorchcodec`` cannot find a compatible
    FFmpeg shared library, so we decode the waveform ourselves with
    ``soundfile`` / ``librosa`` and feed the processor a numpy array.
    """
    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path, always_2d=False)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
    except Exception:
        import librosa

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio = audio.astype(np.float32, copy=False)

    if target_sr and sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(
            np.float32, copy=False
        )
        sr = target_sr

    return audio, sr


def _processor_target_sampling_rate(processor):
    fe = getattr(processor, "feature_extractor", None)
    sr = getattr(fe, "sampling_rate", None) if fe is not None else None
    return int(sr) if sr else 16000


def _build_inputs_via_processor(processor, audio_path, prompt_text):
    """Build model inputs while sidestepping torchcodec/FFmpeg.

    We always decode the audio into a numpy array first (using
    soundfile / librosa) and pass that array to the processor, because
    GLM-ASR's processor would otherwise call ``torchcodec`` on the file
    path -- and torchcodec frequently fails to load its bundled FFmpeg
    shared libraries in containerised environments.
    """
    target_sr = _processor_target_sampling_rate(processor)
    audio_array, _sr = _load_audio_array(audio_path, target_sr)

    if hasattr(processor, "apply_transcription_request"):
        try:
            return processor.apply_transcription_request(audio_array)
        except Exception as e:
            logger.warning(
                "apply_transcription_request(array) failed (%s); falling back.",
                e,
            )
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning("apply_chat_template(array) failed: %s", e)
    raise RuntimeError(
        "Loaded processor does not expose a usable transcription API "
        "(apply_transcription_request / apply_chat_template)."
    )


def _move_to_device(inputs, device, dtype):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device, dtype=dtype)
        except TypeError:
            try:
                return inputs.to(device)
            except Exception:
                pass
    if isinstance(inputs, dict):
        moved = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    moved[k] = v.to(device=device, dtype=dtype)
                else:
                    moved[k] = v.to(device=device)
            else:
                moved[k] = v
        return moved
    return inputs


def _input_id_length(inputs):
    if isinstance(inputs, dict):
        ids = inputs.get("input_ids")
    else:
        ids = getattr(inputs, "input_ids", None)
    if ids is None:
        return 0
    try:
        return int(ids.shape[1])
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=list(_LANG_PROMPT.keys()) + [None],
    )
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    if args.language:
        prompt_text = _LANG_PROMPT[args.language]
    else:
        prompt_text = _DEFAULT_PROMPT
    logger.info("GLM-ASR prompt_text=%r", prompt_text)

    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device == "cpu":
        dtype = torch.float32

    attn_impl = _select_attn_implementation(args.device, args.attn_implementation)
    logger.info(
        "Loading GLM-ASR model from %s (dtype=%s, attn=%s, device=%s)",
        args.path, args.dtype, attn_impl, args.device,
    )

    processor = AutoProcessor.from_pretrained(args.path, trust_remote_code=True)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.path,
            dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(
            "AutoModelForSeq2SeqLM failed (%s); trying AutoModelForCausalLM.", e
        )
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.path,
            dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    model = model.to(args.device)
    model.eval()
    logger.info("GLM-ASR model loaded successfully on %s", args.device)

    while True:
        try:
            prompt = input()
        except EOFError:
            break

        try:
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contain '->' but got {}".format(prompt),
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
                print(f"{prefix}Error: audio file not found: {audio_path}", flush=True)
                continue

            kwargs = payload.get("kwargs", {}) or {}
            max_new_tokens = int(kwargs.pop("max_new_tokens", args.max_new_tokens))
            temperature = float(kwargs.pop("temperature", 0.0))
            top_p = float(kwargs.pop("top_p", 1.0))
            num_beams = int(kwargs.pop("num_beams", 1))
            do_sample = bool(kwargs.pop("do_sample", temperature > 0))
            repetition_penalty = float(kwargs.pop("repetition_penalty", 1.0))
            no_repeat_ngram_size = int(kwargs.pop("no_repeat_ngram_size", 0))

            start_time = time.time()
            inputs = _build_inputs_via_processor(processor, audio_path, prompt_text)
            inputs = _move_to_device(inputs, args.device, dtype)
            input_length = _input_id_length(inputs)

            generation_config = {"max_new_tokens": max_new_tokens}
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

            with torch.inference_mode():
                if isinstance(inputs, dict):
                    output_ids = model.generate(**inputs, **generation_config)
                else:
                    output_ids = model.generate(**dict(inputs), **generation_config)

            if input_length > 0 and output_ids.shape[1] > input_length:
                gen_ids = output_ids[:, input_length:]
            else:
                gen_ids = output_ids

            decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
            raw_text = decoded[0] if decoded else ""
            text = _LEADING_ROLE_RE.sub("", raw_text or "").strip()
            text = _clean_asr_text(text)
            text = _collapse_repeats(text)

            elapsed = time.time() - start_time
            logger.info("ASR done in %.2fs, text length=%d", elapsed, len(text))

            result = json.dumps(
                {"content": text, "raw_text": raw_text}, ensure_ascii=False
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
                    logger.debug("close signal not received within 30s, will emit again")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)


if __name__ == "__main__":
    main()
