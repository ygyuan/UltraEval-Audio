import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to OmniVoice model checkpoint or HuggingFace repo id.",
    )
    parser.add_argument(
        "--num_step",
        type=int,
        default=32,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Scale for Classifier-Free Guidance.",
    )
    parser.add_argument(
        "--t_shift",
        type=float,
        default=0.1,
        help="Shift t to smaller ones if t_shift < 1.0",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading OmniVoice model from {args.path}")

    model = OmniVoice.from_pretrained(
        args.path, device_map=device, dtype=torch.float16
    )
    logger.info("OmniVoice model successfully loaded")

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
            x = json.loads(prompt[anchor + 2 :])

            text = x.pop("text", None)
            ref_audio = x.pop("prompt_audio", None)
            ref_text = x.pop("prompt_text", None)
            language = x.pop("language", None)

            if not text:
                print(f"{prefix}Error: 'text' field is required", flush=True)
                continue

            with torch.no_grad():
                # Record start time for RTF calculation
                start_time = time.time()

                audios = model.generate(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    num_step=args.num_step,
                    guidance_scale=args.guidance_scale,
                    t_shift=args.t_shift,
                    denoise=True,
                    postprocess_output=True,
                )

                # Record end time
                end_time = time.time()
                inference_time = end_time - start_time

                # OmniVoice outputs at 24000 Hz sample rate
                sample_rate = model.sampling_rate

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as f:
                    torchaudio.save(f.name, audios[0], sample_rate)
                    output_path = f.name

                # Return different format based on ENABLE_RTF setting
                if enable_rtf == 1:
                    # Calculate audio duration
                    audio_duration = audios[0].shape[-1] / sample_rate
                    # Calculate RTF (Real Time Factor)
                    rtf = (
                        inference_time / audio_duration
                        if audio_duration > 0
                        else 0
                    )
                    result = json.dumps({"audio": output_path, "RTF": rtf})
                    logger.info(
                        f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, "
                        f"audio: {audio_duration:.2f}s)"
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
                    print(
                        "not found close signal, will emit again", flush=True
                    )

        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
