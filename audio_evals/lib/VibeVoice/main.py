"""
VibeVoice TTS main script for isolated subprocess execution.

This script loads the VibeVoice model and handles voice clone TTS requests
via stdin/stdout communication with the parent process.

Reference: https://github.com/microsoft/VibeVoice
"""

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
import numpy as np
import librosa

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to VibeVoice model"
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
        default="cuda",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="CFG scale for generation",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of DDPM inference steps",
    )
    args = parser.parse_args()

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading VibeVoice model from {args.path} with dtype={args.dtype}")

    # Load processor
    processor = VibeVoiceProcessor.from_pretrained(args.path)

    # Load model with flash attention, fallback to sdpa
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.path,
            torch_dtype=dtype,
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        logger.info("Loaded with flash_attention_2")
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2: {e}, falling back to sdpa")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.path,
            torch_dtype=dtype,
            device_map=args.device,
            attn_implementation="sdpa",
        )

    model.eval()

    # Use SDE solver for better quality
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type="sde-dpmsolver++",
        beta_schedule="squaredcos_cap_v2"
    )
    model.set_ddpm_inference_steps(num_steps=args.num_steps)

    logger.info("VibeVoice model loaded successfully")

    # Enable RTF tracking from environment variable
    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain '->'  , but got {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2:])

            text = x.pop("text", None)
            prompt_audio = x.pop("prompt_audio", None)
            prompt_text = x.pop("prompt_text", None)
            language = x.pop("language", None)

            if not text:
                print(f"{prefix}Error: 'text' field is required", flush=True)
                continue

            # Format text as VibeVoice expects: "Speaker 1: {text}"
            formatted_text = f"Speaker 1: {text}"

            # Prepare voice samples (reference audio for voice cloning)
            voice_samples = None
            if prompt_audio and os.path.exists(prompt_audio):
                voice_samples = [[prompt_audio]]
            else:
                voice_samples = None

            # Record start time for RTF calculation
            start_time = time.time()

            with torch.no_grad():
                # Process inputs
                inputs = processor(
                    text=[formatted_text],
                    voice_samples=voice_samples,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move tensors to device
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(args.device)

                # Generate audio
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=args.cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    **x,
                )

            # Record end time
            end_time = time.time()
            inference_time = end_time - start_time

            # Save output audio
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    processor.save_audio(
                        outputs.speech_outputs[0],
                        output_path=f.name,
                    )
                    output_path = f.name

                # Return result with optional RTF
                if enable_rtf == 1:
                    speech_output = outputs.speech_outputs[0]
                    if isinstance(speech_output, torch.Tensor):
                        audio_samples = speech_output.shape[-1]
                    else:
                        audio_samples = len(speech_output)
                    sample_rate = 24000  # VibeVoice default sample rate
                    audio_duration = audio_samples / sample_rate
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
            else:
                print(f"{prefix}Error: No audio output generated", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
