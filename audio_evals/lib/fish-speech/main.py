"""
fish-speech Inference Server for UltraEval-Audio
"""

import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

# Add current directory first to import local fishspeech_inference.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Add fish-speech repo to path (cloned via pre_command)
FISH_SPEECH_REPO_DIR = "third_party/fish-speech"
sys.path.insert(1, FISH_SPEECH_REPO_DIR)

import torch
import torchaudio

from fishspeech_inference import load_models, generate_speech, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fish-speech Inference Server")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to fish-speech checkpoint directory"
    )
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Compile model with torch.compile"
    )
    parser.add_argument(
        "--half", action="store_true", default=False, help="Use float16 instead of bfloat16"
    )
    args = parser.parse_args()

    precision = torch.half if args.half else torch.bfloat16

    logger.info(f"Loading fish-speech models from {args.ckpt_dir}")
    models = load_models(
        ckpt_dir=args.ckpt_dir,
        repo_dir=FISH_SPEECH_REPO_DIR,
        precision=precision,
        compile=args.compile,
    )
    logger.info("fish-speech models loaded successfully")

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    print("fish-speech server started. Waiting for input...", flush=True)

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

            text = x["text"]
            prompt_audio = x["prompt_audio"]
            prompt_text = x.get("prompt_text", "")
            seed = x.get("seed", 0)

            # Optional generation parameters
            max_new_tokens = x.get("max_new_tokens", 1024)
            top_p = x.get("top_p", 0.8)
            top_k = x.get("top_k", 30)
            temperature = x.get("temperature", 0.8)
            chunk_length = x.get("chunk_length", 200)

            with torch.no_grad():
                start_time = time.time()

                # Generate speech
                audio, sample_rate = generate_speech(
                    models=models,
                    text=text,
                    prompt_audio_path=prompt_audio,
                    prompt_text=prompt_text,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    chunk_length=chunk_length,
                    compile=args.compile,
                )

                end_time = time.time()
                inference_time = end_time - start_time

                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    # audio shape: (T,), need to add channel dim
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    torchaudio.save(f.name, audio, sample_rate)
                    output_path = f.name

                if enable_rtf == 1:
                    audio_duration = audio.shape[-1] / sample_rate
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
            import traceback

            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
