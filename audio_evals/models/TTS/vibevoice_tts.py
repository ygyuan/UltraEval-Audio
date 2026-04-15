"""
VibeVoice TTS model wrapper for UltraEval-Audio.

Supports voice cloning via reference audio for zero-shot TTS evaluation.

Reference: https://github.com/microsoft/VibeVoice
"""

import json
import logging
import os
import select
import time
import uuid
from typing import Dict, Any, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/VibeVoice/main.py")
class VibeVoiceTTS(OfflineModel):
    """
    VibeVoice TTS model for voice cloning evaluation.

    VibeVoice is a frontier long conversational text-to-speech model
    from Microsoft, supporting high-quality voice cloning with
    AR + diffusion architecture.

    Reference: https://github.com/microsoft/VibeVoice
    """

    def __init__(
        self,
        path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cfg_scale: float = 1.3,
        num_steps: int = 10,
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize VibeVoice TTS model.

        Args:
            path: Model path or HuggingFace model ID (e.g., "microsoft/VibeVoice-1.5b")
            dtype: Model dtype - "float16", "bfloat16", or "float32"
            device: Device to run on, e.g., "cuda"
            cfg_scale: CFG scale for generation (default: 1.3)
            num_steps: Number of DDPM inference steps (default: 10)
            sample_params: Additional sampling parameters
        """
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "dtype": dtype,
            "device": device,
            "cfg_scale": str(cfg_scale),
            "num_steps": str(num_steps),
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Send text and audio prompt to the VibeVoice subprocess
        and return the output file path.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'text' (str): The text to synthesize
                - 'prompt_audio' (str): Path to reference audio file for voice cloning
                - 'prompt_text' (str, optional): Transcript of reference audio
                - 'language' (str, optional): Language of the text
            **kwargs: Additional keyword arguments

        Returns:
            str: The output audio file path

        Raises:
            RuntimeError: If the TTS script returns an error or times out
        """
        text = prompt.get("text")
        prompt_audio = prompt.get("prompt_audio")
        prompt_text = prompt.get("prompt_text", None)
        language = prompt.get("language", None)

        if not isinstance(text, str):
            raise TypeError(
                f"Expected 'text' in prompt to be string, but got: {type(text)}"
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Construct JSON request payload
        request_data = {
            "text": text,
        }
        if prompt_audio:
            request_data["prompt_audio"] = prompt_audio
        if prompt_text:
            request_data["prompt_text"] = prompt_text
        if language:
            request_data["language"] = language

        request_json = json.dumps(request_data, ensure_ascii=False)
        request = f"{prefix}{request_json}" + "\n"

        logger.debug(f"Sending request to VibeVoice process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 180
            )
            if xlist:
                raise RuntimeError(
                    "VibeVoice stdin broken (select reported error)"
                )
            if not wlist:
                raise TimeoutError("Timeout waiting for VibeVoice stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("VibeVoice process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(
                f"Error writing to VibeVoice process stdin: {e}"
            )

        # Receive response
        max_wait_time = 600  # Longer timeout for TTS generation
        start_time = time.time()
        response_line = None

        while time.time() - start_time < max_wait_time:
            try:
                reads, _, xlist = select.select(
                    [self.process.stdout, self.process.stderr],
                    [],
                    [self.process.stdout, self.process.stderr],
                    1.0,
                )
                if xlist:
                    raise RuntimeError(
                        "VibeVoice stdout/stderr broken (select reported error)"
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            # Classify subprocess stderr by content level
                            if any(
                                kw in error_output
                                for kw in [
                                    "INFO",
                                    "DEBUG",
                                    "Loading",
                                    "Building",
                                    "loading",
                                    "building",
                                    "done",
                                    "loaded",
                                    "%|",
                                    "it/s]",
                                ]
                            ):
                                logger.debug(
                                    f"VibeVoice stderr: {error_output}"
                                )
                            elif any(
                                kw in error_output
                                for kw in [
                                    "WARNING",
                                    "FutureWarning",
                                    "UserWarning",
                                    "DeprecationWarning",
                                    "deprecated",
                                    "pkg_resources",
                                    "Setting `pad_token_id`",
                                ]
                            ):
                                logger.warning(
                                    f"VibeVoice stderr: {error_output}"
                                )
                            else:
                                logger.error(
                                    f"VibeVoice stderr: {error_output}"
                                )
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix):]
                                self.process.stdin.write(f"{prefix}close" + "\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                raise RuntimeError(
                                    f"VibeVoice failed: {result}"
                                )
                            else:
                                logger.info(result)
            except Exception as e:
                if "VibeVoice" in str(e):
                    raise
                raise RuntimeError(
                    f"Error reading from VibeVoice process: {e}"
                )

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from VibeVoice process "
                f"for request {uid}"
            )
