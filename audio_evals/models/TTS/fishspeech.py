import json
import logging
import select
import uuid
import time
from typing import Dict, Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import os

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/fish-speech/main.py",
    pre_command="mkdir -p ./third_party && ([ ! -d './third_party/fish-speech' ] && "
    "git clone https://github.com/fishaudio/fish-speech.git ./third_party/fish-speech) || true && pwd",
)
class fishspeech(OfflineModel):
    """
    Client for interacting with the isolated fish-speech processing script.

    fish-speech is a high-quality text-to-speech synthesis system based on
    DualAR Transformer + DAC codec, supporting zero-shot voice cloning.
    """

    def __init__(
        self,
        ckpt_dir: str,
        use_cache: bool = True,
        use_phoneme: bool = False,
        compile: bool = False,
        half: bool = False,
        sample_params: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize the fish-speech client.

        Args:
            ckpt_dir: Path to fish-speech checkpoint directory
                (e.g., "fishaudio/s2-pro")
            use_cache: Not used for fish-speech (kept for API compatibility)
            use_phoneme: Not used for fish-speech (kept for API compatibility)
            compile: Whether to compile model with torch.compile
            half: Use float16 instead of bfloat16
            sample_params: Additional sampling parameters
        """
        if not os.path.exists(ckpt_dir):
            ckpt_dir = self._download_model(ckpt_dir)
        self.command_args = {
            "ckpt_dir": ckpt_dir,
        }

        if compile:
            self.command_args["compile"] = ""
        if half:
            self.command_args["half"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Send text and audio prompt to the fish-speech server script
        and return the raw JSON response string containing the output file path.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'text' (str): The text to synthesize
                - 'prompt_audio' (str): Path to reference audio file
                - 'prompt_text' (str, optional): Transcript of reference audio
            **kwargs: Additional keyword arguments

        Returns:
            str: The raw JSON response string from the server script

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
        if not isinstance(prompt_audio, str):
            raise TypeError(
                f"Expected 'prompt_audio' in prompt to be string, but got: {type(prompt_audio)}"
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Construct JSON request payload
        request_data = {
            "text": text,
            "prompt_audio": prompt_audio,
        }
        if prompt_text:
            request_data["prompt_text"] = prompt_text

        request_json = json.dumps(request_data, ensure_ascii=False)
        request = f"{prefix}{request_json}\n"

        logger.debug(f"Sending request to fish-speech process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 180
            )
            if xlist:
                raise RuntimeError("fish-speech stdin broken (select reported error)")
            if not wlist:
                raise TimeoutError("Timeout waiting for fish-speech stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("fish-speech process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(f"Error writing to fish-speech process stdin: {e}")

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
                        "fish-speech stdout/stderr broken (select reported error)"
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            # Classify subprocess stderr by content level
                            if any(kw in error_output for kw in ["INFO", "DEBUG", "Loading checkpoint", "Building prefix dict", "loading fst", "done", "%|"]):
                                logger.debug(f"fish-speech stderr: {error_output}")
                            elif any(kw in error_output for kw in ["WARNING", "FutureWarning", "UserWarning", "DeprecationWarning", "pkg_resources is deprecated"]):
                                logger.warning(f"fish-speech stderr: {error_output}")
                            else:
                                logger.error(f"fish-speech stderr: {error_output}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}close\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                raise RuntimeError(f"fish-speech failed: {result}")
                            else:
                                logger.info(result)
            except Exception as e:
                raise RuntimeError(f"Error reading from fish-speech process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from fish-speech process for request {uid}"
            )
