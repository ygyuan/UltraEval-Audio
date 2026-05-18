"""Fun-Audio-Chat isolated subprocess inference script.

This script is launched in an isolated Python environment via
``audio_evals.isolate.isolated``. It receives JSON-formatted inference
requests from stdin, runs the Fun-Audio-Chat model, and writes the
result (also JSON) back to stdout.

Request protocol (one request per line):
    <uuid>-> {"prompt": [...], "task": "s2t" | "s2s", ...}

Response protocol:
    <uuid>-> {"text": "..."}                    # for s2t / asr / s2tt / emotion
    <uuid>-> {"text": "...", "audio": "/path"}  # for s2s (spoken QA)
"""

import argparse
import json
import os
import select
import sys
import tempfile
import traceback

import librosa
import torch


def _ensure_funaudiochat_importable(repo_root: str) -> None:
    """Put the Fun-Audio-Chat repo root on ``sys.path`` so that
    ``funaudiochat`` / ``utils`` / ``third_party/CosyVoice`` can be imported.
    """
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cosyvoice_path = os.path.join(repo_root, "third_party", "CosyVoice")
    if os.path.isdir(cosyvoice_path) and cosyvoice_path not in sys.path:
        sys.path.insert(0, cosyvoice_path)
    matcha_path = os.path.join(
        repo_root, "third_party", "CosyVoice", "third_party", "Matcha-TTS"
    )
    if os.path.isdir(matcha_path) and matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)


def _extract_audio_and_text(prompt):
    """Extract audio path and text instruction from a single-turn prompt.

    Prompt shape (produced by ``Prompt.load`` in the eval framework):
        [{"role": "user", "contents": [
            {"type": "audio", "value": "/path/to.wav"},
            {"type": "text",  "value": "..."},
        ]}]
    """
    audio_path = None
    text_parts = []
    system_parts = []
    for msg in prompt:
        role = msg.get("role", "user")
        contents = msg.get("contents") or msg.get("content") or []
        if isinstance(contents, str):
            if role == "system":
                system_parts.append(contents)
            else:
                text_parts.append(contents)
            continue
        for c in contents:
            ctype = c.get("type", "text")
            cvalue = c.get("value", c.get(ctype, ""))
            if ctype == "audio":
                audio_path = cvalue
            elif ctype == "text":
                if role == "system":
                    system_parts.append(cvalue)
                else:
                    text_parts.append(cvalue)
    text = "\n".join(t for t in text_parts if t).strip()
    system = "\n".join(t for t in system_parts if t).strip()
    return audio_path, text, system


_ASR_KEYWORDS = (
    "transcribe",
    "transcription",
    "转写",
    "转录",
    "纯文本",
    "语音识别",
    "语音转文字",
)

_S2TT_KEYWORDS = (
    "translation",
    "translate",
    "翻译",
)

_EMOTION_KEYWORDS = (
    "emotion",
    "情感",
    "情绪",
    "surprise",
    "anger",
    "neutral",
    "joy",
    "sadness",
    "fear",
    "disgust",
)


def _guess_task(text: str, explicit_task: str = None) -> str:
    """Decide which generation mode to use."""
    if explicit_task:
        return explicit_task.lower().strip()
    if not text:
        # Audio-only prompt => spoken QA (needs to output speech for ASR).
        return "s2s"
    low = text.lower()
    if any(k in low for k in _ASR_KEYWORDS) or any(
        k in text for k in _ASR_KEYWORDS if not k.isascii()
    ):
        return "s2t"
    if any(k in low for k in _S2TT_KEYWORDS) or any(
        k in text for k in _S2TT_KEYWORDS if not k.isascii()
    ):
        return "s2t"
    if any(k in low for k in _EMOTION_KEYWORDS) or any(
        k in text for k in _EMOTION_KEYWORDS if not k.isascii()
    ):
        return "s2t"
    # Text instruction + audio => generic audio understanding, still only
    # requires text output.
    return "s2t"


class FunAudioChatWorker:
    def __init__(self, model_path: str, tts_model_path: str, device: str,
                 tts_out_dir: str):
        self.model_path = model_path
        self.tts_model_path = tts_model_path
        self.device = device
        self.tts_out_dir = tts_out_dir
        os.makedirs(self.tts_out_dir, exist_ok=True)

        # Register FunAudioChat classes with AutoConfig / AutoProcessor / Auto*
        from funaudiochat.register import register_funaudiochat
        register_funaudiochat()

        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

        config = AutoConfig.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

        # Lazy-load CosyVoice detokenizer only if the user requests S2S.
        self._cosyvoice_model = None

    # ------------------------------------------------------------------
    # Lazy CosyVoice detokenizer
    # ------------------------------------------------------------------
    def _get_cosyvoice(self):
        if self._cosyvoice_model is not None:
            return self._cosyvoice_model
        # We import lazily because loading CosyVoice is expensive and only
        # needed for S2S tasks.
        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore

        token_hop_len = 25 * 30
        cosyvoice3 = CosyVoice3(
            self.tts_model_path,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )
        cosyvoice3.model.flow.decoder.estimator.static_chunk_size = 2 * token_hop_len
        self._cosyvoice_model = cosyvoice3
        return cosyvoice3

    def _token2wav(self, audio_token_ids):
        """Convert audio tokens to a waveform using CosyVoice3."""
        import torchaudio
        import uuid as _uuid

        cosyvoice_model = self._get_cosyvoice()

        # Filter tokens to the valid CosyVoice codebook range.
        tokens = [t for t in audio_token_ids if 0 <= t < 6561]
        if not tokens:
            return None, None

        token_hop_len = 25 * 30
        pre_lookahead_len = 3

        # Default spk embedding (Chinese female).
        spk_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "init_model", "FunAudioLLM",
            "Fun-Audio-Chat", "utils", "new_spk2info.pt",
        )
        spk_path = os.path.abspath(spk_path)
        if not os.path.isfile(spk_path):
            # Fallback: try repo-relative path used by the upstream script.
            spk_path = "utils/new_spk2info.pt"
        embedding = torch.load(spk_path, map_location="cpu")["中文女"]["embedding"]

        # Split tokens into 30s chunks (following upstream token2wav).
        speech_list = []
        tokens_list = []
        time_step = 0
        while time_step * 25 < len(tokens):
            start = time_step * 25
            end = min((time_step + 30) * 25, len(tokens))
            tokens_list.append(tokens[start:end])
            time_step += 30
        if len(tokens_list) > 1 and len(tokens_list[-1]) < 50:
            last_segment = tokens_list.pop()
            second_last_segment = tokens_list.pop()
            merged = second_last_segment + last_segment
            split_point = len(merged) // 2
            tokens_list.append(merged[:split_point])
            tokens_list.append(merged[split_point:])

        for token_segment in tokens_list:
            this_uuid = str(_uuid.uuid4())
            cosyvoice_model.model.hift_cache_dict[this_uuid] = None
            token_offset = 0
            for i in range(0, len(token_segment), token_hop_len):
                this_token = torch.tensor(
                    token_segment[: token_offset + token_hop_len + pre_lookahead_len]
                ).view(1, -1)
                finalize = this_token.shape[1] == len(token_segment)
                this_speech = cosyvoice_model.model.token2wav(
                    this_token,
                    torch.zeros(1, 0, dtype=torch.int32),
                    torch.zeros(1, 0, 80),
                    embedding,
                    token_offset,
                    this_uuid,
                    stream=False,
                    finalize=finalize,
                    speed=1.0,
                )
                speech_list.append(this_speech)
                token_offset += token_hop_len
            del cosyvoice_model.model.hift_cache_dict[this_uuid]

        speech = torch.concat(speech_list, dim=1)
        out_path = os.path.join(self.tts_out_dir, f"{_uuid.uuid4().hex}.wav")
        torchaudio.save(out_path, speech.cpu(), cosyvoice_model.sample_rate)
        return out_path, cosyvoice_model.sample_rate

    # ------------------------------------------------------------------
    # Text generation (S2T)
    # ------------------------------------------------------------------
    def run_s2t(self, audio_path: str, text_instruction: str, system_prompt: str):
        """Generate text-only response (ASR / S2TT / emotion / audio QA)."""
        from utils.constant import DEFAULT_S2T_PROMPT, AUDIO_TEMPLATE

        if not audio_path:
            raise ValueError("S2T task requires an audio input")

        # Build conversation in the official format.
        audio = [librosa.load(audio_path, sr=16000)[0]]
        sys_msg = system_prompt if system_prompt else DEFAULT_S2T_PROMPT
        if text_instruction:
            user_msg = AUDIO_TEMPLATE + "\n" + text_instruction
        else:
            user_msg = AUDIO_TEMPLATE
        conversation = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # Configure decoding: text-only (disable speech head).
        self.model.sp_gen_kwargs.update({
            "text_greedy": True,
            "disable_speech": True,
        })

        tokenized = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=tokenized,
            audio=audio,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        with torch.inference_mode():
            generate_ids, _ = self.model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text = self.processor.decode(generate_ids[0], skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------
    # Speech generation (S2S, spoken QA)
    # ------------------------------------------------------------------
    def run_s2s(self, audio_path: str, text_instruction: str, system_prompt: str):
        """Generate both text and speech for spoken dialogue / QA tasks."""
        from utils.constant import (
            DEFAULT_SP_GEN_KWARGS,
            DEFAULT_S2M_GEN_KWARGS,
            SPOKEN_S2M_PROMPT,
            AUDIO_TEMPLATE,
        )

        if not audio_path:
            raise ValueError("S2S task requires an audio input")

        audio = [librosa.load(audio_path, sr=16000)[0]]
        sys_msg = system_prompt if system_prompt else SPOKEN_S2M_PROMPT
        if text_instruction:
            user_msg = AUDIO_TEMPLATE + "\n" + text_instruction
        else:
            user_msg = AUDIO_TEMPLATE
        conversation = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # Configure decoding for joint text+speech generation.
        sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        sp_gen_kwargs["text_greedy"] = True
        gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
        gen_kwargs["max_new_tokens"] = 2048
        self.model.sp_gen_kwargs.update(sp_gen_kwargs)

        tokenized = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=tokenized,
            audio=audio,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        with torch.inference_mode():
            generate_ids, audio_ids = self.model.generate(**inputs, **gen_kwargs)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        # Try to convert audio tokens to wav. If CosyVoice fails, still return
        # the text response so downstream ASR post-processing can fall back.
        audio_out_path = None
        try:
            audio_token_ids = audio_ids[0].tolist()
            audio_out_path, _ = self._token2wav(audio_token_ids)
        except Exception as err:  # pragma: no cover - depends on CosyVoice deps
            print(
                f"[funaudio_chat] token2wav failed: {err}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

        return text, audio_out_path


def _run_request(worker: FunAudioChatWorker, payload):
    """Dispatch a single request and return a JSON-serialisable result dict."""
    if isinstance(payload, list):
        prompt = payload
        explicit_task = None
        extra_kwargs = {}
    elif isinstance(payload, dict):
        prompt = payload.get("prompt") or payload.get("conversation") or []
        explicit_task = payload.get("task")
        extra_kwargs = {
            k: v for k, v in payload.items()
            if k not in ("prompt", "conversation", "task")
        }
    else:
        raise ValueError(f"Unsupported payload type: {type(payload)!r}")

    audio_path, text_instruction, system_prompt = _extract_audio_and_text(prompt)
    task = _guess_task(text_instruction, explicit_task=explicit_task)

    print(
        f"[funaudio_chat] dispatch task={task} audio={audio_path} "
        f"has_instruction={bool(text_instruction)}",
        file=sys.stderr,
        flush=True,
    )

    if task in ("s2t", "asr", "s2tt", "emotion", "understanding"):
        text = worker.run_s2t(audio_path, text_instruction, system_prompt)
        return {"text": text}

    # Default to S2S (spoken dialogue / QA): generate speech and return its path
    # so the post-processing pipeline (extract_audio + speech2text) works.
    text, audio_out = worker.run_s2s(audio_path, text_instruction, system_prompt)
    result = {"text": text}
    if audio_out and os.path.exists(audio_out) and os.path.getsize(audio_out) > 0:
        result["audio"] = audio_out
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to Fun-Audio-Chat-8B model directory",
    )
    parser.add_argument(
        "--tts_model_path", type=str, required=True,
        help="Path to Fun-CosyVoice3-0.5B-2512 model directory",
    )
    parser.add_argument(
        "--repo_root", type=str, default=None,
        help="Path to the Fun-Audio-Chat source repo (needed for imports).",
    )
    parser.add_argument(
        "--tts_out_dir", type=str,
        default=os.path.join(tempfile.gettempdir(), "funaudio_chat_s2s"),
        help="Directory where generated TTS wav files will be stored.",
    )
    config = parser.parse_args()

    repo_root = config.repo_root
    if repo_root is None:
        # Default to the repo checked out alongside this project.
        repo_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "init_model", "FunAudioLLM", "Fun-Audio-Chat",
        )
    _ensure_funaudiochat_importable(repo_root)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    worker = FunAudioChatWorker(
        model_path=config.model_path,
        tts_model_path=config.tts_model_path,
        device=device,
        tts_out_dir=config.tts_out_dir,
    )

    # Signal that the model has been loaded. The parent process is waiting for
    # the exact "Model loaded" marker below.
    print("Model loaded from checkpoint: {}".format(config.model_path), flush=True)

    while True:
        try:
            line = input()
            anchor = line.find("->")
            if anchor == -1:
                print(
                    "Error: invalid request (missing '->'): {}".format(line),
                    flush=True,
                )
                continue
            prefix = line[:anchor].strip() + "->"
            raw_body = line[anchor + 2:]
            if raw_body.strip() in ("close", "{}close".format(prefix)):
                # Stale close signal leaked to outer loop — ignore.
                continue
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError as err:
                print(
                    "Error: invalid JSON payload: {} ({})".format(err, raw_body[:200]),
                    flush=True,
                )
                continue

            try:
                result = _run_request(worker, payload)
            except Exception as err:
                traceback.print_exc()
                print("Error: inference failed: {}".format(err), flush=True)
                continue

            retry = 3
            response = prefix + json.dumps(result, ensure_ascii=False)
            while retry:
                retry -= 1
                print(response, flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print(
                    "not found close signal, will emit again",
                    file=sys.stderr,
                    flush=True,
                )
        except EOFError:
            break
        except Exception as err:  # pragma: no cover - defensive
            traceback.print_exc()
            print("Error: {}".format(err), flush=True)


if __name__ == "__main__":
    main()
