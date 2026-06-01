"""
PilotTTS subprocess entry point.

This file is launched in an isolated virtual environment by
``audio_evals.isolate.isolated``. It loads the PilotTTS InferenceEngine once
(zero-shot voice cloning) and then serves stdin/stdout JSON requests until it
receives a "close" command for the current request.

Reference: third_party/PilotTTS/inference.py
"""

import argparse
import copy
import json
import logging
import os
import select
import sys
import tempfile
import time
import warnings

# Suppress noisy FutureWarning / UserWarning that PilotTTS / torch / transformers
# emit during model loading (e.g. torch.load weights_only=False FutureWarning,
# torch.nn.utils.weight_norm deprecation, ttsfrd missing, etc.). They are
# harmless but pollute the parent process's stderr classification.
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ---------------------------------------------------------------------------
# Hijack sys.stdout to keep the subprocess <-> parent JSON protocol clean.
#
# PilotTTS bundles CosyVoice3, which prints various messages straight to
# stdout during initialization (e.g. "/root/cuda12.8.bashrc",
# "Now using node v12.18.3", "failed to import ttsfrd, use wetext instead",
# "streaming False", ...). The same is true for shell startup output coming
# from `source envs/PilotTTS/bin/activate`.
#
# All such noise pollutes the stdout pipe that the parent process uses to
# read the synthesis response (`<uuid>-><wav_path>`). We therefore redirect
# sys.stdout (and the underlying fd 1) to fd 2 (stderr) right at startup,
# and keep the *original* stdout file descriptor for the protocol layer.
# ---------------------------------------------------------------------------
_PROTOCOL_STDOUT_FD = os.dup(1)
_PROTOCOL_STDOUT = os.fdopen(_PROTOCOL_STDOUT_FD, "w", buffering=1)
# Redirect fd 1 -> fd 2 so that anyone writing to /dev/stdout (or
# `print()` after this point) ends up on stderr. The parent classifies
# stderr lines as debug/warn so they no longer look like protocol replies.
os.dup2(2, 1)
sys.stdout = sys.stderr


def _send_protocol(line: str) -> None:
    """Write ``line`` (without trailing newline) to the protected stdout fd
    so that the parent process can parse it as a protocol response."""
    _PROTOCOL_STDOUT.write(line + "\n")
    _PROTOCOL_STDOUT.flush()


import torch
import torchaudio
import yaml

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


def _prepend_syspath(path):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _build_runtime_config(args):
    """Read PilotTTS YAML config and override every relative path with the
    absolute paths supplied from the parent process so that the subprocess
    does not depend on the cwd."""
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    config["checkpoint_path"] = args.checkpoint
    config.setdefault("model", {})
    config.setdefault("tokenizer", {})
    config.setdefault("vocoder", {})
    config.setdefault("spk_embedding", {})

    if args.qwen_path:
        config["model"]["pretrain_path"] = args.qwen_path
    if args.w2v_path:
        config["model"]["w2v_path"] = args.w2v_path
    if args.w2v_stats_path:
        config["model"]["w2v_stats_path"] = args.w2v_stats_path
    if args.cosyvoice_path:
        config["vocoder"]["model_dir"] = args.cosyvoice_path
        # campplus.onnx ships inside the CosyVoice model dir
        config["spk_embedding"]["campplus_path"] = os.path.join(
            args.cosyvoice_path, "campplus.onnx"
        )
    if args.tokenizer_path:
        config["tokenizer"]["path"] = args.tokenizer_path
    if args.language:
        config["language"] = args.language

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", required=True,
                        help="Path to the third_party/PilotTTS repo")
    parser.add_argument("--config_path", required=True,
                        help="Absolute path to a PilotTTS YAML config")
    parser.add_argument("--checkpoint", required=True,
                        help="Absolute path to pilot_tts.pt or "
                             "pilot_tts_instruct.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--qwen_path", default="")
    parser.add_argument("--w2v_path", default="")
    parser.add_argument("--w2v_stats_path", default="")
    parser.add_argument("--cosyvoice_path", default="")
    parser.add_argument("--tokenizer_path", default="")
    args = parser.parse_args()

    # Make ``import pilot_voice``/``cosyvoice``/``matcha`` work
    repo_path = os.path.abspath(args.repo_path)
    _prepend_syspath(repo_path)
    _prepend_syspath(os.path.join(repo_path, "third_party"))
    _prepend_syspath(os.path.join(repo_path, "third_party", "Matcha-TTS"))

    # PilotTTS engine.py expects to be importable from `repo_path`. Run
    # the subprocess from the repo dir so the engine's downstream relative
    # asset lookups (`assert/...`, etc.) keep working.
    try:
        os.chdir(repo_path)
    except OSError as e:
        logger.warning(f"Failed to chdir to repo_path: {e}")

    from pilot_voice.engine import InferenceEngine

    config = _build_runtime_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Loading PilotTTS: ckpt={config['checkpoint_path']} "
        f"config={args.config_path} device={device}"
    )
    engine = InferenceEngine(config, device)
    sample_rate = config.get("vocoder", {}).get("sample_rate", 24000)
    logger.info(f"PilotTTS loaded successfully, sample_rate={sample_rate}")

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    while True:
        try:
            prompt = input()
        except EOFError:
            logger.info("stdin closed, exiting")
            break

        anchor = prompt.find("->")
        if anchor == -1:
            _send_protocol(
                f"Error: Invalid conversation format, must contain '->', "
                f"but got {prompt}"
            )
            continue

        prefix = prompt[:anchor].strip() + "->"
        try:
            x = json.loads(prompt[anchor + 2:])
        except json.JSONDecodeError as e:
            _send_protocol(f"{prefix}Error: invalid json payload: {e}")
            continue

        try:
            text = x.get("text")
            prompt_audio = x.get("prompt_audio")
            language = x.get("language") or config.get("language", "zh")
            emotion = x.get("emotion")

            if not text:
                _send_protocol(f"{prefix}Error: 'text' field is required")
                continue
            if not prompt_audio or not os.path.exists(prompt_audio):
                _send_protocol(
                    f"{prefix}Error: prompt_audio not found: {prompt_audio}"
                )
                continue

            # Wrap the text with an emotion tag the same way the official
            # demo / inference.py does for the instruct checkpoint.
            target_text = (
                f"<|{emotion}|>{text}<|/{emotion}|>" if emotion else text
            )

            start_time = time.time()
            with torch.no_grad():
                _, speech = engine.synthesize(
                    prompt_audio, target_text, language=language
                )
            inference_time = time.time() - start_time

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            torchaudio.save(output_path, speech.cpu(), sample_rate=sample_rate)

            if enable_rtf == 1:
                audio_duration = speech.shape[-1] / float(sample_rate)
                rtf = (
                    inference_time / audio_duration
                    if audio_duration > 0 else 0.0
                )
                result = json.dumps({"audio": output_path, "RTF": rtf})
                logger.info(
                    f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, "
                    f"audio: {audio_duration:.2f}s)"
                )
            else:
                result = output_path

            retry = 10
            while retry:
                retry -= 1
                _send_protocol(f"{prefix}{result}")
                rlist, _, _ = select.select([sys.stdin], [], [], 5)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                logger.warning("not found close signal, will emit again")

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _send_protocol(f"{prefix}Error: {str(e)}")


if __name__ == "__main__":
    main()
