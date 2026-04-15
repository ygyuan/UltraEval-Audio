import json
import logging
import os
import random
import tempfile
from typing import Dict, List, Union

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence

logger = logging.getLogger(__name__)


def cut_moshi_greetings(audio_file, output_file):
    # 读取音频文件
    audio = AudioSegment.from_file(audio_file)  # 替换为你的音频文件路径

    # 检测沉默部分
    silence_parts = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)

    # 打印沉默部分
    print("检测到的沉默部分（单位：毫秒）：", audio_file, silence_parts)

    # 如果有沉默部分，将其剪裁掉
    if silence_parts:
        non_silent_audio = audio[:1]  # 删除开场白
        # 大于1s的沉默部分，用1s填充
        for i in range(1, len(silence_parts)):
            non_silent_audio += AudioSegment.silent(1000)
            non_silent_audio += audio[silence_parts[i - 1][1]:silence_parts[i][0]]
        non_silent_audio += AudioSegment.silent(1000)
        non_silent_audio += audio[silence_parts[-1][1]:]  # 保留最后一个沉默后部分
    else:
        non_silent_audio = audio  # 如果没有检测到沉默，保留原始音频

    # 保存处理后的音频
    assert output_file.endswith(".wav"), "输出文件名必须以.wav结尾"
    non_silent_audio.export(output_file, format="wav")  # 替换为目标文件名和格式


def save_audio_response(response, output_file, sample_rate, volume=1.0, cut_greeting=False):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        audio_tensor = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                text = data["text"]
                token_id = data['audio']
                if "sampleRate" in data:
                    sample_rate = data["sampleRate"]
                audio_tensor.append(np.array(token_id))
        audio_tensor = np.concatenate(audio_tensor, axis=0)
        # Handle both float32 waveform (range [-1.0, 1.0]) and int16 formats
        if audio_tensor.dtype in (np.float32, np.float64) or (np.abs(audio_tensor).max() <= 1.0 + 1e-6 and len(audio_tensor) > 0):
            audio_tensor = np.clip(audio_tensor, -1.0, 1.0).astype(np.float32)
            audio_tensor *= float(volume)
            audio_tensor = np.clip(audio_tensor, -32768, 32767).astype(np.int16)
        else:
            audio_tensor *= int(volume)
            audio_tensor = np.array(audio_tensor, dtype=np.int16)
        sf.write(output_file, audio_tensor, sample_rate)
        if cut_greeting:
            cut_moshi_greetings(output_file, output_file)
        return output_file, text
    else:
        response_text = response.text.strip()
        if response_text:
            raise Exception(f"下载失败，状态码: {response.status_code}, 响应内容: {response_text[:500]}")
        raise Exception(f"下载失败，状态码: {response.status_code}")


def prepare_audio_file(audio_file, target_sample_rate=24000):
    _, file_extension = os.path.splitext(audio_file)
    if file_extension.lower() == ".wav":
        audio = AudioSegment.from_file(audio_file)
        if audio.frame_rate == target_sample_rate and audio.channels == 1:
            return None, audio_file
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    audio.export(temp_file.name, format="wav")
    return temp_file.name, temp_file.name


class GLM4Voice(APIModel):
    def __init__(
        self,
        url: Union[str, List[str]],
        sr: int,
        volume: float = 1.0,
        cut_greeting: bool = False,
        sample_params: Dict[str, any] = None,
        *args,
        env_path: str = None,
        requirements_path: str = None,
        asr_backend: str = "glm_native",
        asr_model_path: str = "openai/whisper-large-v3",
        asr_env_path: str = "envs/whisper",
        asr_requirements_path: str = "audio_evals/lib/whisper/requirements.txt",
        **kwargs,
    ):
        super().__init__(True, sample_params)
        self.url = url if isinstance(url, list) else [url]
        self.sr = sr
        self.volume = volume
        self.cut_greeting = cut_greeting
        self.env_path = env_path
        self.requirements_path = requirements_path
        self.asr_backend = asr_backend
        self.asr_model_path = asr_model_path
        self.asr_env_path = asr_env_path
        self.asr_requirements_path = asr_requirements_path

    def _looks_like_asr_prompt(self, text_prompt: str) -> bool:
        normalized = text_prompt.strip().lower()
        if not normalized:
            return False
        asr_keywords = [
            "transcribe",
            "transcription",
            "请识别",
            "识别这段",
            "语音内容",
            "转写",
            "听写",
        ]
        return any(keyword in normalized for keyword in asr_keywords)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = ""
        text_prompt = ""
        for content in prompt:
            if content["role"] != "user":
                continue
            for line in content["contents"]:
                if line["type"] == "audio" and not audio_file:
                    audio_file = line["value"]
                elif line["type"] == "text":
                    text_prompt += line["value"]

        if self._looks_like_asr_prompt(text_prompt):
            if self.asr_backend != "glm_native":
                raise ValueError(
                    f"Unsupported GLM4Voice ASR backend: {self.asr_backend}. "
                    "Please use asr_backend='glm_native' to run native GLM-4-Voice token-based ASR."
                )
            logger.info(
                "Routing GLM4Voice ASR prompt to the native GLM-4-Voice token pipeline via the adapter server."
            )

        temp_audio_file = None
        try:
            temp_audio_file, normalized_audio_file = prepare_audio_file(audio_file)
            audio_base64 = get_base64_from_file(normalized_audio_file)
            headers = {
                'Content-Type': 'application/json'
            }
            data = {
                'prompt': text_prompt,
                'audio': audio_base64
            }
            url = random.choice(self.url)
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio, text = save_audio_response(response, f.name, self.sr, self.volume, self.cut_greeting)
                return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
