import argparse
import json
import select
import sys

import librosa
import torch
from transformers import AutoTokenizer, AutoModel


device = "cuda"


def load_audio(audio_path, sr=24000):
    """Load and resample audio to target sample rate."""
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    return audio


def encode_audio(audio_tokenizer, audio_path, device="cuda"):
    """Encode audio file into discrete tokens using MiMo-Audio-Tokenizer."""
    audio = load_audio(audio_path, sr=24000)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        audio_codes = audio_tokenizer.encode(audio_tensor)
    # audio_codes shape: (batch, n_codebooks, time)
    # Use the first codebook for the LLM input
    return audio_codes[0, 0, :].cpu().tolist()


def build_chat_input(tokenizer, audio_tokenizer, messages, device="cuda"):
    """Build input_ids for MiMo-Audio-7B-Instruct from chat messages.

    Format follows the Qwen2 chat template:
    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    <|im_start|>user\n<|sosp|>...audio_tokens...<|eosp|>text<|im_end|>\n
    <|im_start|>assistant\n
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sosp_id = tokenizer.convert_tokens_to_ids("<|sosp|>")
    eosp_id = tokenizer.convert_tokens_to_ids("<|eosp|>")
    nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    input_ids = []

    # System message
    system_ids = tokenizer.encode("system\nYou are a helpful assistant.", add_special_tokens=False)
    input_ids.extend([im_start_id] + system_ids + [im_end_id, nl_id])

    for msg in messages:
        role = msg["role"]
        contents = msg.get("contents", msg.get("content", []))
        if isinstance(contents, str):
            contents = [{"type": "text", "value": contents}]

        role_ids = tokenizer.encode(f"{role}\n", add_special_tokens=False)
        input_ids.extend([im_start_id] + role_ids)

        for content in contents:
            ctype = content.get("type", "text")
            cvalue = content.get("value", content.get(ctype, ""))

            if ctype == "text":
                text_ids = tokenizer.encode(cvalue, add_special_tokens=False)
                input_ids.extend(text_ids)
            elif ctype == "audio":
                audio_tokens = encode_audio(audio_tokenizer, cvalue, device=device)
                input_ids.append(sosp_id)
                input_ids.extend(audio_tokens)
                input_ids.append(eosp_id)

        input_ids.extend([im_end_id, nl_id])

    # Generation prompt
    assistant_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    input_ids.extend([im_start_id] + assistant_ids)

    return torch.tensor([input_ids], dtype=torch.long).to(device)


def try_load_model_with_mimo_audio(model_path, tokenizer_path, device="cuda"):
    """Try to load model using MiMo-Audio package first, fallback to transformers."""
    try:
        # Try importing MiMo-Audio package
        from mimo_audio.model import MiMoAudioModel as MiMoModel
        from mimo_audio.tokenizer import MiMoAudioTokenizer

        print("Using MiMo-Audio package for model loading", file=sys.stderr, flush=True)
        audio_tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path).to(device).eval()
        model = MiMoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, audio_tokenizer, "mimo_audio"
    except ImportError:
        pass

    # Fallback: load with transformers trust_remote_code
    print("Loading with transformers (trust_remote_code=True)", file=sys.stderr, flush=True)
    audio_tokenizer = AutoModel.from_pretrained(
        tokenizer_path, trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    return model, tokenizer, audio_tokenizer, "transformers"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to MiMo-Audio-7B-Instruct model"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to MiMo-Audio-Tokenizer"
    )
    config = parser.parse_args()

    model, tokenizer, audio_tokenizer, backend = try_load_model_with_mimo_audio(
        config.model_path, config.tokenizer_path, device=device
    )
    print("Model loaded from checkpoint: {}".format(config.model_path), flush=True)

    while True:
        try:
            prompt = input()

            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contain ->, but {}".format(prompt),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            conversation = json.loads(prompt[anchor + 2:])

            # Build input
            input_ids = build_chat_input(
                tokenizer, audio_tokenizer, conversation, device=device
            )

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode generated part only
            generated_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            retry = 3
            while retry:
                retry -= 1
                print(prefix + json.dumps({"text": text}), flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error:" + str(e))
