import argparse
import json
import logging
import os
import select
import sys
import numpy as np
import torch
import soundfile as sf
import scipy
import whisper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_TARGET_SR = 16000


def _resample_to_target_sr(wav: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample 1-D float32 mono waveform to ``_TARGET_SR`` (16 kHz).

    Tries soxr -> scipy.signal.resample_poly -> librosa in order. None of
    these shell out to ``ffmpeg`` or use ``torchaudio.load``.
    """
    if src_sr == _TARGET_SR:
        return wav.astype(np.float32, copy=False)

    # Backend 1: soxr (fast, high-quality; bundled with librosa>=0.10)
    try:
        import soxr

        return soxr.resample(
            wav.astype(np.float32, copy=False), src_sr, _TARGET_SR
        ).astype(np.float32)
    except Exception:
        pass

    # Backend 2: scipy.signal.resample_poly
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(src_sr), _TARGET_SR)
        up = _TARGET_SR // g
        down = int(src_sr) // g
        return resample_poly(
            wav.astype(np.float32, copy=False), up, down
        ).astype(np.float32)
    except Exception:
        pass

    # Backend 3: librosa (final fallback)
    import librosa

    return librosa.resample(
        wav.astype(np.float32, copy=False),
        orig_sr=int(src_sr),
        target_sr=_TARGET_SR,
    ).astype(np.float32)


def _decode_to_16k_mono_array(src_path: str) -> np.ndarray:
    """Decode ``src_path`` to a 1-D float32 mono numpy waveform at 16 kHz.

    Avoids OpenAI-Whisper's default ``ffmpeg``-based ``load_audio`` (which
    fails with ``FileNotFoundError`` when ``ffmpeg`` is not on PATH) by
    decoding in-process via libsndfile / librosa.
    """
    # --- backend 1: libsndfile (wav/flac/ogg/opus/etc.) ---
    try:
        wav, sr = sf.read(src_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:  # multi-channel -> mono
            wav = wav.mean(axis=1)
        return _resample_to_target_sr(
            wav.astype(np.float32, copy=False), int(sr)
        )
    except Exception as e_sf:
        print(
            f"[cv3-whisper] soundfile decode failed for {src_path}: {e_sf}",
            file=sys.stderr,
            flush=True,
        )

    # --- backend 2: librosa (handles non-libsndfile containers) ---
    import librosa

    wav, _ = librosa.load(src_path, sr=_TARGET_SR, mono=True)
    return wav.astype(np.float32, copy=False)


def _load_whisper_model(path: str):
    """Load OpenAI Whisper checkpoint.

    cv3 whisper subprocess shares the same physical GPU with vllm-omni
    serving Higgs Audio v3 (which has already reserved most of the GPU
    via ``gpu_memory_utilization=0.9``). The default fp32 load may
    still OOM the worker (visible as exit code -9 from the shell). In
    that case we transparently fall back to CPU so the eval still
    completes (just slower).

    NOTE: We deliberately do NOT cast the model to fp16 here. OpenAI
    Whisper's ``transcribe(..., fp16=True)`` internally keeps the model
    weights in fp32 and casts only the mel input / activations to fp16
    (its custom LayerNorm relies on the weights staying fp32). Casting
    the model itself with ``.half()`` triggers
    ``RuntimeError: expected scalar type Float but found Half`` inside
    LayerNorm.
    """
    use_cuda = torch.cuda.is_available()

    # Allow explicit override via env var (e.g. CV3_WHISPER_DEVICE=cpu).
    forced_device = os.environ.get("CV3_WHISPER_DEVICE", "").strip().lower()
    if forced_device == "cpu":
        logger.info("CV3_WHISPER_DEVICE=cpu, loading whisper on CPU")
        return whisper.load_model(path, device="cpu"), "cpu", False

    if not use_cuda:
        return whisper.load_model(path, device="cpu"), "cpu", False

    try:
        model = whisper.load_model(path, device="cuda:0")
        return model, "cuda:0", True
    except torch.cuda.OutOfMemoryError as oom:
        logger.warning(
            "CUDA OOM while loading whisper on cuda:0 (%s); retrying on CPU. "
            "This is usually caused by another process (e.g. vllm-omni) "
            "having already reserved most of the GPU memory.",
            oom,
        )
        torch.cuda.empty_cache()
        return whisper.load_model(path, device="cpu"), "cpu", False
    except RuntimeError as exc:
        # Torch may surface OOM through the generic RuntimeError type as
        # well (e.g. "CUDA error: out of memory"). Retry on CPU.
        if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
            logger.warning(
                "CUDA error while loading whisper on cuda:0 (%s); retrying on CPU.",
                exc,
            )
            torch.cuda.empty_cache()
            return whisper.load_model(path, device="cpu"), "cpu", False
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to Whisper model")
    config = parser.parse_args()

    # Initialize model
    model, device, fp16 = _load_whisper_model(config.path)
    model.eval()
    logger.info(
        f"Using Whisper model from: {config.path} on device: {device} (fp16={fp16})"
    )
    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])
            # Process input
            logger.info(f"Received input: {x}")

            # Decode to numpy ndarray in-process so OpenAI-Whisper's
            # transcribe() takes the ndarray branch and bypasses its
            # default ffmpeg-based ``load_audio`` (which would fail with
            # FileNotFoundError on hosts without ffmpeg installed).
            audio_input = _decode_to_16k_mono_array(x["audio"])

            result = model.transcribe(
                audio_input,
                language=x.get("generate_kwargs", {}).get("language", "english"),
                fp16=fp16,
            )
            transcription = result["text"].strip()
            result = {"text": transcription}
            retry = 3
            while retry:
                print(f"{prefix}{result['text']}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
