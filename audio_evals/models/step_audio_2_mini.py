"""
Step-Audio-2-mini model integration using transformers inference.

This implementation follows the official Step-Audio2 transformers flow
instead of starting a separate vLLM server.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import librosa
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

AUDIO_TOKEN_TEXT_THRESHOLD = 151688
AUDIO_TOKEN_OFFSET = 151696
AUDIO_CHUNK_SIZE = 16000 * 25


def _mel_filters(n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    return torch.from_numpy(
        librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
    )


def _load_audio(file_path: str, target_rate: int = 16000) -> torch.Tensor:
    waveform, _ = librosa.load(file_path, sr=target_rate, mono=True)
    return torch.from_numpy(waveform).float()


def _log_mel_spectrogram(
    audio: Union[torch.Tensor, str],
    n_mels: int = 128,
    padding: int = 479,
) -> torch.Tensor:
    if not torch.is_tensor(audio):
        audio = _load_audio(audio)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def _compute_token_num(max_feature_len: int) -> int:
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding = 1
    kernel_size = 3
    stride = 2
    return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1


def _padding_mels(data: List[torch.Tensor]):
    feats_lengths = torch.tensor([item.size(1) - 2 for item in data], dtype=torch.int32)
    feats = [item.t() for item in data]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return padded_feats.transpose(1, 2), feats_lengths


class StepAudio2Mini(OfflineModel):
    """Step-Audio-2-mini model using in-process transformers inference."""

    def __init__(
        self,
        model_path: str,
        env_path: Optional[str] = None,
        requirements_path: Optional[str] = None,
        gpu_id: Optional[Union[int, str]] = None,
        start_port: int = 9999,
        tensor_parallel_size: int = 4,
        max_model_len: int = 16384,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float = 0.85,
        startup_timeout: int = 600,
        extract_thinking: bool = True,
        speech: bool = False,
        sample_params: Dict[str, Any] = None,
    ):
        del env_path
        del requirements_path
        del start_port
        del tensor_parallel_size
        del max_num_seqs
        del gpu_memory_utilization
        del startup_timeout

        if not os.path.exists(model_path):
            model_path = self._download_model_from_modelscope(model_path)
        if model_path.endswith("/"):
            model_path = model_path[:-1]

        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.max_model_len = max_model_len
        self.extract_thinking = extract_thinking
        self.speech = speech
        self.device = self._resolve_device(gpu_id)
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        logger.info(
            "Loading Step-Audio-2-mini from %s on %s with dtype=%s",
            self.model_path,
            self.device,
            self.dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
        if isinstance(eot_token_id, int) and eot_token_id >= 0:
            self.tokenizer.eos_token = "<|EOT|>"
            self.model.config.eos_token_id = eot_token_id
            self.eos_token_id = eot_token_id
        else:
            self.eos_token_id = self.tokenizer.eos_token_id

        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _resolve_device(gpu_id: Optional[Union[int, str]]) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        if gpu_id is None:
            return "cuda"
        if isinstance(gpu_id, str):
            gpu_id = gpu_id.split(",")[0].strip()
        return f"cuda:{gpu_id}"

    def _convert_prompt_to_messages(self, prompt: PromptStruct) -> List[Dict[str, Any]]:
        messages = []

        # Inject system prompt if not already present (required by Step-Audio-2)
        has_system = any(item.get("role") == "system" for item in prompt)
        if not has_system:
            messages.append({"role": "system", "content": "You are a helpful assistant."})

        for item in prompt:
            role = item["role"]
            contents = item.get("contents", [])

            if role == "user":
                role = "human"

            if not contents:
                messages.append({"role": role, "content": None})
                continue

            content_list = []
            for content in contents:
                content_type = content.get("type")
                value = content.get("value")

                if content_type == "text":
                    content_list.append({"type": "text", "text": value})
                elif content_type == "audio":
                    content_list.append({"type": "audio", "audio": value})

            if len(content_list) == 1 and content_list[0].get("type") == "text":
                messages.append({"role": role, "content": content_list[0]["text"]})
            else:
                messages.append({"role": role, "content": content_list})

        messages.append({"role": "assistant", "content": None})
        return messages

    def _apply_chat_template(self, messages: List[Dict[str, Any]]):
        results = []
        mels = []

        for message in messages:
            role = message["role"]
            content = message.get("content")

            if role == "user":
                role = "human"

            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                if message.get("eot", True):
                    text_with_audio += "<|EOT|>"
                results.append(text_with_audio)
            elif isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(item["text"])
                    elif item["type"] == "audio":
                        audio = _load_audio(item["audio"])
                        for i in range(0, audio.shape[0], AUDIO_CHUNK_SIZE):
                            mel = _log_mel_spectrogram(audio[i : i + AUDIO_CHUNK_SIZE])
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                if message.get("eot", True):
                    results.append("<|EOT|>")
            elif content is None:
                results.append(f"<|BOT|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")

        return results, mels

    def _build_generate_inputs(self, messages: List[Dict[str, Any]]):
        prompt_parts, mels = self._apply_chat_template(messages)

        prompt_ids = []
        for part in prompt_parts:
            if isinstance(part, str):
                token_ids = self.tokenizer(
                    text=part,
                    return_tensors="pt",
                    padding=True,
                )["input_ids"]
                prompt_ids.append(token_ids)
            elif isinstance(part, list):
                prompt_ids.append(torch.tensor([part], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")

        input_ids = torch.cat(prompt_ids, dim=-1).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        wavs = None
        wav_lens = None
        if mels:
            wavs, wav_lens = _padding_mels(mels)
            wavs = wavs.to(device=self.device)
            wav_lens = wav_lens.to(self.device)

        return {
            "input_ids": input_ids,
            "wavs": wavs,
            "wav_lens": wav_lens,
            "attention_mask": attention_mask,
        }, input_ids.shape[-1]

    def _normalize_generation_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        generation_kwargs = dict(kwargs)

        if "max_tokens" in generation_kwargs and "max_new_tokens" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = generation_kwargs.pop("max_tokens")
        generation_kwargs.setdefault("max_new_tokens", 2048)

        # Disable KV cache: the custom StepAudio2ForCausalLM.forward() does not
        # propagate past_key_values to the inner Qwen2Model, so the HF generate()
        # loop must re-run the full sequence at every step.
        generation_kwargs.setdefault("use_cache", False)

        stop_token_ids = generation_kwargs.pop("stop_token_ids", None)
        if stop_token_ids and "eos_token_id" not in generation_kwargs:
            generation_kwargs["eos_token_id"] = (
                stop_token_ids if len(stop_token_ids) > 1 else stop_token_ids[0]
            )
        generation_kwargs.setdefault("eos_token_id", self.eos_token_id)
        generation_kwargs.setdefault(
            "pad_token_id",
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id,
        )

        stop = generation_kwargs.pop("stop", None)
        if stop and "stop_strings" not in generation_kwargs:
            if isinstance(stop, Sequence) and all(isinstance(item, str) for item in stop):
                generation_kwargs["stop_strings"] = list(stop)

        generation_kwargs.pop("request_timeout", None)
        generation_kwargs.pop("timeout", None)

        if "do_sample" not in generation_kwargs and generation_kwargs.get("temperature") not in (None, 0, 1.0):
            generation_kwargs["do_sample"] = True

        return generation_kwargs

    # Pattern to match language tags like <中文>, <英文>, <日文>, <韩文>, <粤语>, etc.
    _LANG_TAG_RE = re.compile(r"<[^<>]{1,10}>")

    def _extract_response(self, text: str) -> str:
        if not text:
            return text
        if self.extract_thinking:
            text = text.split("</think>")[-1].strip()
        # Remove language tags produced by Step-Audio-2 (e.g. <中文>)
        text = self._LANG_TAG_RE.sub("", text).strip()
        return text

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        messages = self._convert_prompt_to_messages(prompt)
        generate_inputs, prompt_length = self._build_generate_inputs(messages)
        generation_kwargs = self._normalize_generation_kwargs(kwargs)

        logger.info("Calling %s with transformers inference...", self.model_name)
        logger.debug(
            "Built messages: %s...",
            json.dumps(messages, ensure_ascii=False)[:500],
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **generate_inputs,
                generation_config=GenerationConfig(**generation_kwargs),
                tokenizer=self.tokenizer,
            )

        output_token_ids = outputs[0, prompt_length:].tolist()
        # Remove trailing eos token (same as official: outputs[0, prompt_length:-1])
        if output_token_ids and output_token_ids[-1] == self.eos_token_id:
            output_token_ids = output_token_ids[:-1]

        output_text_tokens = [
            token_id for token_id in output_token_ids if token_id < AUDIO_TOKEN_TEXT_THRESHOLD
        ]
        output_audio_tokens = [
            token_id - AUDIO_TOKEN_OFFSET
            for token_id in output_token_ids
            if token_id >= AUDIO_TOKEN_OFFSET
        ]

        output_text = self.tokenizer.decode(
            output_text_tokens,
            skip_special_tokens=False,
        )
        text_result = self._extract_response(output_text) if output_text else ""

        if not self.speech:
            return text_result

        result = {"text": text_result}
        if output_audio_tokens:
            logger.info("Received %s audio tokens", len(output_audio_tokens))

        return json.dumps(result, ensure_ascii=False)

    def release(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()