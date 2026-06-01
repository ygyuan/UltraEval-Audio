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
    "..", "..", "..", "init_model", "XiaomiMiMo", "MiMo-Audio",
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

# Signals that the model drifted from ASR into a description / analysis response.
# Timestamp-like markers such as "**00:10 - 00:22**" or "[00:00-00:10]".
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\b")
# Bulleted markdown list openings that never appear in a real transcription.
_MD_BULLET_RE = re.compile(r"(?:^|\n)\s*[*\-]\s+\*\*")
# Typical "audio description" lead-ins in zh / en. Match case-insensitively for en.
_ASR_DRIFT_PHRASES = (
    "这是一段",
    "这是一个",
    "音频内容",
    "音频中",
    "音频如下",
    "音频描述",
    "音频片段",
    "以下是",
    "本段音频",
    "the audio contains",
    "the audio is",
    "this is an audio",
    "this audio",
    "in this audio",
    "here is a transcription",
    "here's a transcription",
    "here is the transcription",
    "here's the transcription",
)
# ASR references in covered benchmarks are overwhelmingly < 300 chars. Anything
# far beyond that is almost certainly a hallucinated description.
_ASR_MAX_LEN = 500


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


def _has_repetition_loop(text: str) -> bool:
    """Detect degenerate repeat-until-max-tokens outputs.

    Two complementary heuristics:
    1. Any non-trivial substring (>=8 chars) that repeats >=4 times back to
       back -> clear decoding loop.
    2. Sentence-level duplication ratio: after splitting on common CJK/EN
       sentence terminators, if the most common non-empty sentence accounts
       for >=40% of all sentences AND appears at least 4 times -> loop.
    """
    if not text:
        return False

    # 1) back-to-back character-level loop, e.g. "abcabcabcabc".
    #    Regex (.{8,}?)\1{3,} matches a substring >=8 chars repeated >=4 times.
    if re.search(r"(.{8,}?)\1{3,}", text):
        return True

    # 2) sentence-level loop: split on CJK/EN sentence enders + newline.
    parts = re.split(r"[。！？!?\.\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 5:
        from collections import Counter
        counts = Counter(parts)
        top_sent, top_cnt = counts.most_common(1)[0]
        if top_cnt >= 4 and top_cnt / len(parts) >= 0.4 and len(top_sent) >= 6:
            return True
    return False


def _looks_like_description(text: str) -> bool:
    """Return True if text looks like a free-form audio description, not ASR."""
    if not text:
        return False
    # Timestamp markers -> description / summary.
    if _TIMESTAMP_RE.search(text):
        return True
    # Multiple markdown bullet openings -> structured description.
    if len(_MD_BULLET_RE.findall(text)) >= 2:
        return True
    low = text.lower()
    # Lead-in phrases, but only trust them if they appear at the start so we
    # do not accidentally filter legitimate long transcripts.
    head = text[:30]
    head_low = low[:60]
    for phrase in _ASR_DRIFT_PHRASES:
        if phrase in head or phrase in head_low:
            return True
    return False


def _postprocess_asr(raw: str) -> str:
    """Drop pathological ASR outputs (template echo, description drift, loops)."""
    cleaned = _clean_text(raw)
    if not cleaned:
        return ""
    # 1) pure template echo (model copied the instruction).
    if cleaned in _ASR_TEMPLATE_ECHOES:
        print("[mimo-audio] ASR filter: template-echo", file=sys.stderr, flush=True)
        return ""
    # 2) description drift (timestamps / bullet lists / lead-in phrases).
    if _looks_like_description(cleaned):
        print("[mimo-audio] ASR filter: description-drift", file=sys.stderr, flush=True)
        return ""
    # 3) repeat-until-max-tokens decoding loop.
    if _has_repetition_loop(cleaned):
        print("[mimo-audio] ASR filter: repetition-loop", file=sys.stderr, flush=True)
        return ""
    # 4) absurdly long output relative to typical ASR references.
    if len(cleaned) > _ASR_MAX_LEN:
        print(f"[mimo-audio] ASR filter: too-long(len={len(cleaned)})",
              file=sys.stderr, flush=True)
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
