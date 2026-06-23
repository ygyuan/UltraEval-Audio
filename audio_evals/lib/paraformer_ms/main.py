import argparse
import select
import sys

import numpy as np
import soundfile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


_TARGET_SR = 16000


def _resample_to_target_sr(wav: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample a 1-D float32 mono waveform to ``_TARGET_SR``.

    Tries soxr/scipy/librosa in order. All are pure-Python / native libs and
    do NOT shell out to ``ffmpeg`` or touch ``torchaudio.load``.
    """
    if src_sr == _TARGET_SR:
        return wav.astype(np.float32, copy=False)

    # Backend 1: soxr (used by librosa>=0.10; fast and high-quality)
    try:
        import soxr

        return soxr.resample(wav.astype(np.float32, copy=False), src_sr, _TARGET_SR).astype(np.float32)
    except Exception:
        pass

    # Backend 2: scipy.signal.resample_poly (rational resampling, no compiled deps beyond scipy)
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(src_sr), _TARGET_SR)
        up = _TARGET_SR // g
        down = int(src_sr) // g
        return resample_poly(wav.astype(np.float32, copy=False), up, down).astype(np.float32)
    except Exception:
        pass

    # Backend 3: librosa (audioread/soxr; final fallback)
    import librosa

    return librosa.resample(wav.astype(np.float32, copy=False), orig_sr=int(src_sr), target_sr=_TARGET_SR).astype(np.float32)


def _decode_to_16k_mono_array(src_path: str) -> np.ndarray:
    """Decode ``src_path`` to a 1-D float32 mono numpy waveform at 16 kHz.

    Avoids ``torchaudio.load`` entirely (its >=2.10 backend now requires
    the optional ``torchcodec`` package and silently falls back to a
    system ``ffmpeg`` binary, neither of which is available in this env).

    Backend order:
      1. ``soundfile`` (libsndfile: wav/flac/ogg/opus/etc.)
      2. ``librosa.load`` (audioread / soxr; covers mp3/m4a containers when
         the corresponding audioread dependencies are available).
    """
    # --- backend 1: soundfile (covers wav/flac/ogg natively) ---
    try:
        wav, sr = soundfile.read(src_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _resample_to_target_sr(wav.astype(np.float32, copy=False), int(sr))
    except Exception as e_sf:
        print(
            f"[paraformer_ms] soundfile decode failed for {src_path}: {e_sf}",
            file=sys.stderr,
            flush=True,
        )

    # --- backend 2: librosa.load (handles non-libsndfile containers) ---
    import librosa

    wav, sr = librosa.load(src_path, sr=_TARGET_SR, mono=True)
    return wav.astype(np.float32, copy=False)


def get_model(path, is_streaming=False):
    model_cfg = {
        # "vad_model": "fsmn-vad",
        # "punc_model": "ct-punc-c",
    }
    if is_streaming:
        model_cfg = {}
    print("Loading model from: {}".format(path))
    model = AutoModel(model=path, **model_cfg)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--chunk_size", type=int, default=0, help="Chunk size")
    config = parser.parse_args()
    is_streaming = config.path.endswith("streaming") or config.path.endswith("online")
    chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back = (
        4  # number of chunks to lookback for encoder self-attention
    )
    decoder_chunk_look_back = (
        1  # number of encoder chunks to lookback for decoder cross-attention
    )

    m = get_model(config.path, is_streaming=is_streaming)
    print("Model loaded from checkpoint: {}".format(config.path))

    while True:
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
        try:
            # Decode in-process to a 16 kHz mono float32 ndarray. Passing
            # ndarray (not a file path) to ``m.generate`` makes funasr's
            # ``load_audio_text_image_video`` take the ``np.ndarray`` branch
            # and skip ``torchaudio.load`` entirely (which otherwise tries
            # the missing ``torchcodec`` then a missing system ``ffmpeg``).
            speech = _decode_to_16k_mono_array(prompt[len(prefix) :])

            if is_streaming:
                chunk_stride = chunk_size[1] * 960  # 600ms

                cache = {}
                total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
                transcription = ""
                for i in range(total_chunk_num):
                    speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
                    is_final = i == total_chunk_num - 1
                    res = m.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=is_final,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back,
                        fs=_TARGET_SR,
                    )
                    transcription += "".join([item["text"] for item in res])
            else:
                if config.chunk_size > 0:
                    texts = []
                    sr = _TARGET_SR
                    for start in range(0, len(speech), config.chunk_size * sr):
                        chunk = speech[start : start + config.chunk_size * sr]
                        res = m.generate(input=chunk, batch_size_s=300, fs=_TARGET_SR)
                        if len(res) > 0:
                            text = res[0]["text"]
                            texts.append(text.strip())
                    transcription = "".join(t for t in texts if t)
                else:
                    transcription = m.generate(
                        input=speech, batch_size_s=300, fs=_TARGET_SR
                    )[0]["text"]
            retry = 3
            while retry:
                retry -= 1
                print(
                    "{}{}".format(
                        prefix, rich_transcription_postprocess(transcription)
                    )
                )
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
