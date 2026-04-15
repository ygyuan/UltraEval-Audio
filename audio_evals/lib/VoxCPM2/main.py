import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import torch
import soundfile as sf
from voxcpm import VoxCPM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_numpy(wav):
    # support torch.Tensor / list / numpy
    try:
        import numpy as np
    except Exception:
        np = None

    if "torch" in str(type(wav)):
        wav = wav.detach().cpu().float().numpy()
    elif np is not None and isinstance(wav, np.ndarray):
        wav = wav.astype("float32")
    elif isinstance(wav, list):
        if np is None:
            raise RuntimeError("numpy is required to handle list waveform")
        import numpy as np

        wav = np.asarray(wav, dtype="float32")
    return wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--vc_mode", action="store_true", default=False, help="Enable voice clone mode"
    )
    parser.add_argument("--denoise", action="store_true", help="Enable denoising")
    parser.add_argument(
        "--denoise_path",
        type=str,
        required=False,
        default="./init_model/iic/speech_zipenhancer_ans_multiloss_16k_base",
        help="Path to denoising model",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading VoxCPM2 model from {args.path}, denoise: {args.denoise}")

    # Resolve denoise_path to a local cache path to avoid online hub check.
    # modelscope's pipeline() will try to contact www.modelscope.cn if the
    # model path does not exist locally, even when the model is already cached.
    denoise_model_id = args.denoise_path
    if args.denoise and denoise_model_id and not os.path.exists(denoise_model_id):
        # Try to find the model in modelscope's default cache directory
        from modelscope.utils.file_utils import get_modelscope_cache_dir
        cache_root = os.path.join(get_modelscope_cache_dir(), "hub")
        candidate = os.path.join(cache_root, denoise_model_id)
        if os.path.isdir(candidate):
            denoise_model_id = candidate
            logger.info(f"Resolved denoise model to local cache: {denoise_model_id}")
        else:
            logger.warning(
                f"Denoise model not found in local cache ({candidate}), "
                f"will attempt online download from modelscope."
            )

    model = VoxCPM.from_pretrained(
        args.path, load_denoiser=args.denoise, zipenhancer_model_id=denoise_model_id
    )
    logger.info("VoxCPM2 successfully loaded")

    # Read ENABLE_RTF setting from environment variable, default is 0
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
            print(prompt[anchor + 2 :])
            x = json.loads(prompt[anchor + 2 :])

            with torch.no_grad():
                # Record start time for RTF calculation
                start_time = time.time()

                wav = model.generate(
                    text=x.pop("text"),
                    prompt_wav_path=x.pop("prompt_audio", None),
                    cfg_value=2.0,
                    inference_timesteps=10,
                    **x,
                )
                wav = to_numpy(wav)

                # Record end time
                end_time = time.time()
                inference_time = end_time - start_time

                # VoxCPM2 outputs at 44100 Hz sample rate
                sample_rate = 44100
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, wav, samplerate=sample_rate)
                    output_path = f.name

                # Return different format based on ENABLE_RTF setting
                if enable_rtf == 1:
                    # Calculate audio duration
                    audio_duration = len(wav) / sample_rate
                    # Calculate RTF (Real Time Factor)
                    rtf = inference_time / audio_duration if audio_duration > 0 else 0
                    result = json.dumps({"audio": output_path, "RTF": rtf})
                    logger.info(
                        f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                    )
                else:
                    result = output_path

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
            print(f"Error: {str(e)}", flush=True)
