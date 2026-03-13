# fish-speech inference module for UltraEval-Audio
# GitHub: https://github.com/fishaudio/fish-speech
#
# Modified for UltraEval-Audio: wraps fish-speech's native inference pipeline
# for the eval framework's isolated subprocess architecture.

import logging
import os
import gc

import numpy as np
import torch
import torchaudio

from fish_speech.utils import set_seed
from fish_speech.models.text2semantic.inference import (
    init_model,
    generate_long,
    load_codec_model,
    encode_audio,
    decode_to_audio,
    GenerateResponse,
)

# --- Global Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_models(
    ckpt_dir,
    repo_dir="./third_party/fish-speech",
    precision=None,
    compile=False,
):
    """
    Load all model components for fish-speech inference.

    Args:
        ckpt_dir: Path to checkpoint directory containing model weights
            (e.g., "checkpoints/openaudio-s1-mini")
        repo_dir: Path to fish-speech repo directory
        precision: Torch dtype precision (default: bfloat16)
        compile: Whether to compile the model with torch.compile

    Returns:
        dict with keys: model, decode_one_token, codec, precision
    """
    if precision is None:
        precision = torch.bfloat16

    device = DEVICE

    logger.info(f"Loading DualARTransformer from {ckpt_dir}")
    model, decode_one_token = init_model(
        ckpt_dir, device, precision, compile=compile
    )

    # Setup KV caches
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    logger.info("DualARTransformer loaded and caches initialized")

    # Load DAC codec
    codec_path = os.path.join(ckpt_dir, "codec.pth")
    logger.info(f"Loading DAC codec from {codec_path}")
    codec = load_codec_model(codec_path, device, precision)
    logger.info("DAC codec loaded")

    return {
        "model": model,
        "decode_one_token": decode_one_token,
        "codec": codec,
        "precision": precision,
    }


def generate_speech(
    models,
    text,
    prompt_audio_path=None,
    prompt_text=None,
    seed=0,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    repetition_penalty=1.1,
    chunk_length=200,
    iterative_prompt=True,
    compile=False,
):
    """
    Generate speech from text with optional reference audio for voice cloning.

    Args:
        models: dict returned by load_models()
        text: Text to synthesize
        prompt_audio_path: Path to reference audio file for voice cloning
        prompt_text: Transcript of reference audio
        seed: Random seed for reproducibility
        max_new_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        chunk_length: Chunk length for iterative prompt in bytes
        iterative_prompt: Whether to use iterative prompting
        compile: Whether model was compiled

    Returns:
        tuple: (audio_waveform as torch.Tensor, sample_rate as int)
    """
    model = models["model"]
    decode_one_token = models["decode_one_token"]
    codec = models["codec"]

    device = DEVICE

    # Set random seed
    set_seed(seed)

    # Handle prompt audio encoding
    prompt_tokens_list = None
    prompt_text_list = None

    if prompt_audio_path and os.path.exists(prompt_audio_path):
        logger.info(f"Encoding reference audio: {prompt_audio_path}")
        prompt_tokens = encode_audio(prompt_audio_path, codec, device)
        prompt_tokens_list = [prompt_tokens.cpu()]
        prompt_text_list = [prompt_text] if prompt_text else [" "]
        logger.info(f"Encoded reference audio to VQ codes: {prompt_tokens.shape}")

    # Add speaker tag if not present
    if "<|speaker:" not in text:
        text = f"<|speaker:0|>{text}"

    # Run generation
    codes_list = []

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text_list,
        prompt_tokens=prompt_tokens_list,
    )

    for response in generator:
        if response.action == "sample":
            codes_list.append(response.codes)
            logger.info(f"Generated codes for: {response.text}")
        elif response.action == "next":
            break

    if not codes_list:
        raise RuntimeError("No audio generated, please check the input text.")

    # Merge all codes and decode to audio
    merged_codes = torch.cat(codes_list, dim=1).to(device)
    logger.info(f"Total merged codes shape: {merged_codes.shape}")

    audio = decode_to_audio(merged_codes, codec)
    sample_rate = codec.sample_rate

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return audio.float().cpu(), sample_rate
