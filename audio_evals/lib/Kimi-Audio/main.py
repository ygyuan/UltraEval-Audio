import argparse
import json
import os

# Disable tqdm progress bars and transformers verbose logging at import time.
# tqdm uses \r (carriage return) instead of \n; when its output is captured
# by ``subprocess.PIPE``, the parent process cannot split the stream into
# lines and the OS pipe buffer (default 64KB) fills up, blocking
# ``print(..., flush=True)`` calls on stdout — which is precisely how this
# subprocess signals readiness to the parent. Silence all of it preemptively.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import select
import signal
import sys
import tempfile
import logging
import time
from copy import deepcopy
from kimia_infer.api.kimia import KimiAudio
import soundfile as sf
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLING_PARAMS = {
    "audio_temperature": 0.6,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.05,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.1,
    "text_repetition_window_size": 16,
    "max_new_tokens": 256,
}
SPEECH_SAMPLING_OVERRIDES = {
    "audio_temperature": 0.4,
    "audio_top_k": 8,
    "audio_repetition_penalty": 1.1,
    "max_new_tokens": 48,
}
ALLOWED_SAMPLING_PARAM_KEYS = set(DEFAULT_SAMPLING_PARAMS)


def _get_sampling_params(speech, overrides=None):
    sampling_params = deepcopy(DEFAULT_SAMPLING_PARAMS)
    if speech:
        sampling_params.update(SPEECH_SAMPLING_OVERRIDES)
    if overrides:
        for key, value in overrides.items():
            if key in ALLOWED_SAMPLING_PARAM_KEYS and value is not None:
                sampling_params[key] = value
    return sampling_params


def _handle_fpe(signum, frame):
    """Handle SIGFPE by raising an exception instead of crashing."""
    raise FloatingPointError(f"Caught SIGFPE (signal {signum}) during inference")


def _build_single_turn_messages(audio_path, instruction, speech):
    messages = []
    if instruction:
        messages.append(
            {
                "role": "user",
                "message_type": "text",
                "content": instruction,
            }
        )
    elif not speech:
        messages.append(
            {
                "role": "user",
                "message_type": "text",
                "content": "Please transcribe the following audio:",
            }
        )
    messages.append(
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        }
    )
    return messages


if __name__ == "__main__":
    # Register signal handler for SIGFPE to prevent process crash
    signal.signal(signal.SIGFPE, _handle_fpe)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="Path or HF repo for Kimi-Audio model",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    parser.add_argument(
        "--lazy_detokenizer",
        action="store_true",
        default=False,
        help="Skip detokenizer loading at startup and initialize it only when audio output is actually needed",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Run a single local inference on this audio file and exit",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional user text instruction for one-shot inference",
    )
    parser.add_argument(
        "--output_audio",
        type=str,
        default=None,
        help="Optional output wav path for one-shot speech inference",
    )
    config = parser.parse_args()

    start_time = time.time()
    model = KimiAudio(
        model_path=config.model_path,
        load_detokenizer=not config.lazy_detokenizer,
    )
    end_time = time.time()
    logger.info(f"Model loading took {end_time - start_time:.2f} seconds")
    logger.info(f"Using Kimi-Audio model: {config.model_path}")

    # Signal to parent process that model is ready
    print(f"Model loaded from: {config.model_path}", flush=True)

    if config.audio:
        output_type = "both" if config.speech else "text"
        messages = _build_single_turn_messages(
            audio_path=config.audio,
            instruction=config.instruction,
            speech=config.speech,
        )
        sampling_params = _get_sampling_params(config.speech)
        with torch.no_grad():
            wav, text = model.generate(
                messages, **sampling_params, output_type=output_type
            )

        if config.speech and output_type == "both":
            output_audio_path = config.output_audio
            if not output_audio_path:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    output_audio_path = f.name
            sf.write(
                output_audio_path,
                wav.detach().cpu().view(-1).numpy() if torch.is_tensor(wav) else wav,
                24000,
            )
            print(
                json.dumps(
                    {"text": text, "audio": output_audio_path},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        sys.exit(0)

    if sys.stdin.isatty():
        print(
            "[INFO] Kimi-Audio is ready and waiting for stdin requests in the format '<request_id>->{\"messages\": [...]}'. "
            "For a quick local test, rerun with --audio /path/to.wav [--instruction \"...\"] [--speech --output_audio out.wav].",
            file=sys.stderr,
            flush=True,
        )

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain ->, but {prompt}",
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            # Compatible with UltraEval PromptStruct format, assuming x is PromptStruct
            messages = x["messages"] if "messages" in x else x
            sampling_params = _get_sampling_params(
                config.speech,
                x.get("sampling_params") if isinstance(x, dict) else None,
            )
            force_text_output = bool(x.get("force_text_output")) if isinstance(x, dict) else False
            output_type = "text" if force_text_output else ("both" if config.speech else "text")
            with torch.no_grad():
                wav, text = model.generate(
                    messages, **sampling_params, output_type=output_type
                )

            if config.speech and output_type == "both":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(
                        f.name,
                        (
                            wav.detach().cpu().view(-1).numpy()
                            if torch.is_tensor(wav)
                            else wav
                        ),
                        24000,
                    )
                    retry = 3
                    while retry:
                        print(
                            f"{prefix}{json.dumps({'text': text, 'audio': f.name}, ensure_ascii=False)}",
                            flush=True,
                        )
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == f"{prefix}close":
                                break
                        print("not found close signal, will emit again", flush=True)
                        retry -= 1
            else:
                retry = 3
                while retry:
                    print(
                        f"{prefix}{json.dumps({'text': text}, ensure_ascii=False)}",
                        flush=True,
                    )
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                    print("not found close signal, will emit again", flush=True)
                    retry -= 1

            del wav, text
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        except EOFError:
            print("[INFO] stdin closed, exiting Kimi-Audio worker.", file=sys.stderr, flush=True)
            break
        except KeyboardInterrupt:
            print("[INFO] Interrupted, exiting Kimi-Audio worker.", file=sys.stderr, flush=True)
            break
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
