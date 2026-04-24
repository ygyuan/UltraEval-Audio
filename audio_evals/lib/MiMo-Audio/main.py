import argparse
import json
import os
import re
import select
import sys
import tempfile
import uuid

# Make the offline MiMo-Audio repo importable.
# The official repo layout is:
#   <repo_root>/src/mimo_audio/mimo_audio.py  (uses relative imports)
#   <repo_root>/src/mimo_audio_tokenizer/__init__.py
# The official example uses `from src.mimo_audio.mimo_audio import MimoAudio`,
# which requires <repo_root> to be on sys.path.
_DEFAULT_REPO_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "init_model", "XiaomiMimo", "MiMo-Audio",
)
_REPO_ROOT = os.environ.get("MIMO_AUDIO_REPO", _DEFAULT_REPO_ROOT)
_REPO_ROOT = os.path.abspath(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src.mimo_audio.mimo_audio import MimoAudio  # type: ignore
    # The official ASR prompt uses a randomly-sampled template string from these
    # two lists. On some inputs the model echoes that template verbatim instead
    # of transcribing; we filter those out below.
    from src.mimo_audio.templates import (  # type: ignore
        asr_en_templates,
        asr_zh_templates,
    )
except ImportError as e:
    print(
        "Error: failed to import MimoAudio from offline repo at "
        f"{_REPO_ROOT}. Make sure the MiMo-Audio source repo exists there, "
        "or set env MIMO_AUDIO_REPO to its path. Original error: " + str(e),
        file=sys.stderr,
        flush=True,
    )
    raise


# Where to dump intermediate wav files produced by spoken_dialogue_sft.
_TMP_WAV_DIR = os.environ.get(
    "MIMO_AUDIO_TMP_WAV_DIR",
    os.path.join(tempfile.gettempdir(), "mimo_audio_s2s"),
)
os.makedirs(_TMP_WAV_DIR, exist_ok=True)


_ASR_TEMPLATE_ECHOES = {t.strip() for t in (asr_en_templates + asr_zh_templates)}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_LEADING_ROLE_RE = re.compile(r"^(?:<\|im_start\|>)?\s*assistant\s*\n", flags=re.IGNORECASE)


def _clean_text(text: str) -> str:
    """Strip residual template / role / thinking tokens from model output."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _LEADING_ROLE_RE.sub("", text)
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|empty|>", "")
    text = text.replace("<|eot|>", "")
    text = text.replace("<|eostm|>", "")
    return text.strip()


def _postprocess_asr(raw: str) -> str:
    """Post-process ASR output to drop pathological 'template echo' cases."""
    cleaned = _clean_text(raw)
    if cleaned in _ASR_TEMPLATE_ECHOES:
        # Model copied the ASR instruction instead of transcribing -> fail-soft.
        return ""
    return cleaned


def _extract_audio_and_text(conversation):
    """Extract the audio path and text instruction from a single-turn prompt.

    The conversation comes from `audio_evals.models.mimo_audio.MiMoAudio`
    (see `_parse_role_content` / `_parse_content`), so each item looks like:
        {"role": "user", "content": [
            {"type": "audio", "audio": "/path/to.wav"},
            {"type": "text",  "text": "..."},
        ]}
    """
    audio_path = None
    text = ""
    for msg in conversation:
        contents = msg.get("content", msg.get("contents", []))
        if isinstance(contents, str):
            text = (text + "\n" + contents) if text else contents
            continue
        for c in contents:
            ctype = c.get("type", "text")
            # Accept both {"type": "...", "value": "..."} and
            # {"type": "...", "<type>": "..."}.
            cvalue = c.get(ctype, c.get("value", ""))
            if ctype == "audio":
                audio_path = cvalue
            elif ctype == "text":
                text = (text + "\n" + cvalue) if text else cvalue
    return audio_path, (text or "").strip()


_ASR_KEYWORDS = (
    "transcribe",
    "transcription",
    "转换为纯文本",
    "转录",
    "转写",
    "语音转文字",
    "语音识别",
)


def _guess_task(text, explicit_task=None):
    """Decide which MimoAudio SFT interface to use for the current prompt."""
    if explicit_task:
        return explicit_task.lower().strip()
    if not text:
        # Prompts that only contain an audio turn (e.g. `direct-aqa`) are
        # spoken-QA / spoken-dialogue tasks in this repo.
        return "spoken_dialogue"
    low = text.lower()
    if any(k in low for k in _ASR_KEYWORDS):
        return "asr"
    return "audio_understanding"


def run_inference(model, conversation, explicit_task=None):
    audio_path, text = _extract_audio_and_text(conversation)
    if not audio_path:
        raise ValueError("No audio content found in the prompt")

    task = _guess_task(text, explicit_task=explicit_task)
    print(f"[mimo-audio] dispatch task={task} audio={audio_path}",
          file=sys.stderr, flush=True)

    if task == "asr":
        raw = model.asr_sft(audio_path)
        return _postprocess_asr(raw)

    if task == "spoken_dialogue":
        # For S2S QA / spoken-dialogue evaluation tasks the upstream pipeline
        # post-processes with `extract_audio` + `speech2text`, so we must
        # return the path of a real wav file, not a free-form text reply.
        out_wav = os.path.join(_TMP_WAV_DIR, f"{uuid.uuid4().hex}.wav")
        try:
            model.spoken_dialogue_sft(
                audio_path,
                output_audio_path=out_wav,
                system_prompt=(
                    "You are MiMo-Audio, a friendly AI assistant and your "
                    "response needs to be concise."
                ),
                prompt_speech=None,
                add_history=False,
            )
        except Exception:
            # If S2S generation fails, fall back to text-only dialogue so the
            # evaluator still receives a string instead of crashing the worker.
            fallback = model.speech2text_dialogue_sft(audio_path, thinking=False)
            return _clean_text(fallback)
        return out_wav

    # Default: audio understanding with the given text instruction.
    # Covers S2TT / emotion recognition / generic audio QA.
    return _clean_text(model.audio_understanding_sft(audio_path, text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to MiMo-Audio-7B-Instruct model (local dir or HF id)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to MiMo-Audio-Tokenizer (local dir or HF id)",
    )
    config = parser.parse_args()

    model = MimoAudio(config.model_path, config.tokenizer_path)
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
            payload = json.loads(prompt[anchor + 2:])

            # Payload can either be a bare conversation list, or a dict
            # {"task": "...", "conversation": [...]} to explicitly pick a task.
            explicit_task = None
            if isinstance(payload, dict) and "conversation" in payload:
                conversation = payload["conversation"]
                explicit_task = payload.get("task")
            else:
                conversation = payload

            text = run_inference(model, conversation, explicit_task=explicit_task)
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)

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
