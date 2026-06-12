import json
import logging
import select
import time
import uuid
from typing import Dict, Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import os

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/OmniVoice/main.py",
    pre_command="pip install ./third_party/OmniVoice || pip install git+https://github.com/k2-fsa/OmniVoice.git || true",
)
class OmniVoiceTTS(OfflineModel):
    """
    Client for interacting with the isolated OmniVoice TTS processing script.

    OmniVoice is a state-of-the-art massive multilingual zero-shot
    text-to-speech (TTS) model supporting over 600 languages.
    It supports voice cloning, voice design, and auto voice generation.
    """

    def __init__(
        self,
        path: str,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        sample_params: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the OmniVoice TTS client.

        Args:
            path: Path to OmniVoice model checkpoint or HuggingFace repo id
                (e.g., "k2-fsa/OmniVoice")
            num_step: Number of diffusion steps (default: 32)
            guidance_scale: Scale for Classifier-Free Guidance (default: 2.0)
            t_shift: Shift t to smaller ones if t_shift < 1.0 (default: 0.1)
            sample_params: Additional sampling parameters
        """
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "num_step": str(num_step),
            "guidance_scale": str(guidance_scale),
            "t_shift": str(t_shift),
        }

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Send text and audio prompt to the OmniVoice server script
        and return the output file path.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'text' (str): The text to synthesize
                - 'prompt_audio' (str): Path to reference audio file for voice cloning
                - 'prompt_text' (str, optional): Transcript of reference audio
            **kwargs: Additional keyword arguments

        Returns:
            str: The output audio file path

        Raises:
            RuntimeError: If the TTS script returns an error or times out
            TypeError: If required inputs are not strings
        """
        text = prompt.get("text")
        prompt_audio = prompt.get("prompt_audio")
        prompt_text = prompt.get("prompt_text", None)

        if not isinstance(text, str):
            raise TypeError(
                f"Expected 'text' in prompt to be string, but got: {type(text)}"
            )

        # Make sure the subprocess is alive (auto-restart on crash).
        # Without this, once the OmniVoice subprocess dies (e.g. import
        # error, OOM, segfault), every subsequent sample would write to a
        # broken pipe and fail with a misleading "stdin pipe is broken"
        # error, while the real root-cause stderr is lost until shutdown.
        if hasattr(self, "ensure_process_alive"):
            self.ensure_process_alive()
        elif self.process.poll() is not None:
            raise RuntimeError(
                "OmniVoice subprocess has exited with code "
                f"{self.process.returncode}."
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

        request_json = json.dumps(request_data, ensure_ascii=False)
        request = f"{prefix}{request_json}\n"

        logger.debug(f"Sending request to OmniVoice process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 180
            )
            if xlist:
                raise RuntimeError(
                    "OmniVoice stdin broken (select reported error)"
                )
            if not wlist:
                raise TimeoutError("Timeout waiting for OmniVoice stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("OmniVoice process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(
                f"Error writing to OmniVoice process stdin: {e}"
            )

        # Receive response
        max_wait_time = 300  # Longer timeout for TTS generation
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
                        "OmniVoice stdout/stderr broken (select reported error)"
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
                                    f"OmniVoice stderr: {error_output}"
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
                                ]
                            ):
                                logger.warning(
                                    f"OmniVoice stderr: {error_output}"
                                )
                            else:
                                logger.error(
                                    f"OmniVoice stderr: {error_output}"
                                )
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}close\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                raise RuntimeError(
                                    f"OmniVoice failed: {result}"
                                )
                            else:
                                logger.info(result)
            except Exception as e:
                raise RuntimeError(
                    f"Error reading from OmniVoice process: {e}"
                )

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from OmniVoice process "
                f"for request {uid}"
            )
