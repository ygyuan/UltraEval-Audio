"""
Step-Audio-R1.1 model integration using vLLM with isolated environment.

Step-Audio-R1.1 is a real-time speech model that supports:
- Mind-Paced Speaking (Low Latency)
- Acoustic-Grounded Reasoning (High Intelligence)

Reference: https://huggingface.co/stepfun-ai/Step-Audio-R1.1
"""

import json
import logging
import os
import select
import sys
import threading
import time
from typing import Dict, Any, List

from audio_evals.lib.StepAudio.stepaudior1vllm import StepAudioR1 as StepAudioR1Client

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

@isolated(
    "audio_evals/lib/StepAudio/serve.py",
)
class StepAudioR1(OfflineModel):
    """
    Step-Audio-R1.1 model using vLLM with isolated environment.

    This model starts a vLLM server in an isolated environment and
    communicates via StepAudioR1 HTTP API.

    The server automatically finds an available port and communicates
    it back via stdout.

    Requirements:
        - Customized vLLM from https://github.com/stepfun-ai/vllm (step-audio2-mini branch)
        - Step-Audio-R1.1 model weights
    """

    def __init__(
        self,
        model_path: str,
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
        """
        Initialize Step-Audio-R1.1 model with isolated vLLM server.

        Args:
            model_path: Path to Step-Audio-R1.1 model weights
            start_port: Starting port to search from (default: 9999)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 4)
            max_model_len: Maximum model context length (default: 16384)
            max_num_seqs: Maximum number of sequences (default: 32)
            gpu_memory_utilization: GPU memory utilization (default: 0.85)
            startup_timeout: Timeout for server startup in seconds (default: 600)
            extract_thinking: Whether to extract and remove <think>...</think> blocks
            speech: Whether to enable audio output support (default: False)
            sample_params: Additional sampling parameters
        """
        if not os.path.exists(model_path):
            model_path = self._download_model_from_modelscope(model_path)
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        # Build command args for isolated decorator
        self.command_args = {
            "model_path": model_path,
            "start_port": start_port,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "startup_timeout": startup_timeout,
        }
        self.model_name = model_path.split("/")[-1]
        self.extract_thinking = extract_thinking
        self.speech = speech
        self.port = None
        self.client = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error = None  # Cache initialization failure
        self._stderr_buffer = []

        # Call parent init (this triggers isolated decorator's new_init)
        super().__init__(is_chat=True, sample_params=sample_params)
        # Note: self.process is set by @isolated decorator AFTER this __init__ returns

    def _ensure_initialized(self):
        """Lazy initialization: wait for port and create client on first use (thread-safe)."""
        if self._initialized:
            return

        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            # If a previous initialization attempt failed, raise the cached error
            if self._init_error is not None:
                raise RuntimeError(
                    f"Server initialization previously failed: {self._init_error}"
                )

            try:
                # Wait for server to report port
                self._wait_for_port()

                # Initialize OpenAI client
                api_url = f"http://localhost:{self.port}/v1/chat/completions"
                self.client = StepAudioR1Client(api_url=api_url, model_name=self.model_name)
                logger.info(f"Step-Audio-R1.1 client ready at {api_url}")
                self._initialized = True
            except Exception as e:
                self._init_error = str(e)
                raise

    def _wait_for_port(self):
        """Wait for and read the port number from the server process."""
        logger.info("Waiting for server to report port...")

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line.startswith("PORT:"):
                            self.port = int(line.split(":")[1])
                            logger.info(f"Server started on port {self.port}")
                            return
                        else:
                            logger.debug(f"stdout: {line}")
                if read is self.process.stderr:
                    error_line = self.process.stderr.readline()
                    if error_line:
                        error_line = error_line.rstrip("\n")
                        self._stderr_buffer.append(error_line)
                        self._stderr_buffer = self._stderr_buffer[-200:]
                        logger.info(f"stderr: {error_line}")

            # Check if process has exited
            if self.process.poll() is not None:
                # Collect remaining stderr output for diagnostics
                stderr_output = ""
                try:
                    remaining_stderr = self.process.stderr.read()
                    if remaining_stderr:
                        stderr_output = remaining_stderr.strip()
                except Exception:
                    pass
                exit_code = self.process.returncode
                error_msg = (
                    f"Server process exited before reporting port "
                    f"(exit code: {exit_code})"
                )
                stderr_parts = []
                if self._stderr_buffer:
                    stderr_parts.append("\n".join(self._stderr_buffer))
                if stderr_output:
                    stderr_parts.append(stderr_output)
                if stderr_parts:
                    combined_stderr = "\n".join(stderr_parts)
                    logger.error(f"Server stderr output:\n{combined_stderr}")
                    error_msg += f"\nLast stderr output:\n{combined_stderr[-4000:]}"
                raise RuntimeError(error_msg)

    def _convert_prompt_to_messages(self, prompt: PromptStruct) -> List[Dict]:
        """Convert PromptStruct to StepAudioR1 message format."""
        messages = []

        for item in prompt:
            role = item["role"]
            contents = item.get("contents", [])

            if role == "user":
                role = "human"

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

        messages.append({"role": "assistant", "content": "<think>\n", "eot": False})

        return messages

    def _extract_response(self, text: str) -> str:
        """Extract the actual response, removing <think>...</think> blocks."""
        if not self.extract_thinking or not text:
            return text

        return text.split("</think>")[-1].strip()

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """Perform inference with Step-Audio-R1.1."""
        # Lazy initialization on first call
        self._ensure_initialized()

        # Check process is still running
        if not self.check_process_status():
            raise RuntimeError("vLLM server process has exited")

        # Build messages from prompt
        messages = self._convert_prompt_to_messages(prompt)
        logger.debug(
            f"Built messages: {json.dumps(messages, ensure_ascii=False)[:500]}..."
        )

        # Set default parameters
        api_params = {
            "stop_token_ids": [151665],
        }
        api_params.update(kwargs)

        logger.info(f"Calling {self.model_name} API...")

        full_text = ""
        audio_tokens = []

        timeout_seconds = 180
        # Requests-level timeout: (connect_timeout, read_timeout)
        # - connect_timeout prevents "connection miss" hanging forever
        # - read_timeout prevents "server never returns any bytes" hanging forever
        request_timeout = (10, timeout_seconds)
        start_time = time.time()

        try:
            for _, text, audio in self.client.stream(
                messages, request_timeout=request_timeout, **api_params
            ):
                # Total wall-clock timeout (even if server keeps streaming slowly)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        f"StepAudioR1 stream exceeded {timeout_seconds}s (elapsed: {elapsed:.2f}s)"
                    )

                if text:
                    full_text += text
                if audio:
                    audio_tokens.extend(audio)
        except TimeoutError as e:
            logger.error(f"Timeout during API call: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            raise

        text_result = self._extract_response(full_text) if full_text else ""

        if not self.speech:
            return text_result

        result = {"text": text_result}

        if audio_tokens:
            logger.info(f"Received {len(audio_tokens)} audio tokens")

        return json.dumps(result, ensure_ascii=False)
