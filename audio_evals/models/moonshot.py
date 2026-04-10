from itertools import chain
import fcntl
import json
import logging
import os
import subprocess
import time
import uuid
from copy import deepcopy
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
WRITE_TIMEOUT = 60
READ_POLL_TIMEOUT = 1.0
INFERENCE_TIMEOUT = 1800  # Max total time for a single inference call in speech mode
MODEL_LOAD_TIMEOUT = 1800  # Max time to wait for model loading (includes CUDA compilation and detokenizer init)
MAX_RESTART_ATTEMPTS = 3  # Max number of subprocess restart attempts
DEFAULT_SPEECH_QA_INSTRUCTION = (
    "Please answer the question in the audio briefly and naturally. "
    "Keep the response concise and grounded in the audio content."
)
ALLOWED_SAMPLING_PARAM_KEYS = {
    "audio_temperature",
    "audio_top_k",
    "text_temperature",
    "text_top_k",
    "audio_repetition_penalty",
    "audio_repetition_window_size",
    "text_repetition_penalty",
    "text_repetition_window_size",
    "max_new_tokens",
}


@isolated("audio_evals/lib/Kimi-Audio/main.py")
class KimiAudioModel(OfflineModel):
    def __init__(
        self,
        model_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if model_path == "moonshotai/Kimi-Audio-7B-Instruct" and not os.path.exists(
            model_path
        ):
            model_path = self._download_model(model_path)

        self.command_args = {
            "model_path": model_path,
        }
        if speech:
            self.command_args["speech"] = ""

        self.speech = speech
        self._ready = False
        self._restart_count = 0
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_role_content(self, role_content: Dict):
        assert isinstance(
            role_content["contents"], list
        ), "prompt should be list not string"

        res = []

        for c in role_content["contents"]:
            temp = {
                "role": role_content["role"],
                "message_type": c["type"],
                "content": c["value"],
            }
            res.append(temp)
        return res

    def _prepare_prompt(self, prompt: PromptStruct):
        if not self.speech:
            return prompt

        has_audio = False
        has_text = False
        for role_content in prompt:
            for content in role_content["contents"]:
                content_type = content.get("type")
                if content_type == "audio":
                    has_audio = True
                elif content_type == "text" and str(content.get("value", "")).strip():
                    has_text = True

        if has_audio and not has_text:
            return [
                {
                    "role": "user",
                    "contents": [
                        {
                            "type": "text",
                            "value": DEFAULT_SPEECH_QA_INSTRUCTION,
                        }
                    ],
                },
                *deepcopy(prompt),
            ]
        return prompt

    def _should_force_text_output(self, prompt: PromptStruct) -> bool:
        if not self.speech:
            return False

        if not isinstance(prompt, list) or not prompt:
            return False

        if any(item.get("role") != "user" for item in prompt):
            return False

        has_audio = False
        for item in prompt:
            contents = item.get("contents", [])
            for content in contents:
                if content.get("type") == "audio":
                    has_audio = True
                elif content.get("type") == "text":
                    text = str(content.get("value", "")).strip().lower()
                    if any(keyword in text for keyword in ["speak", "speech", "voice", "read aloud", "audio reply", "say it"]):
                        return False

        return has_audio

    def _collect_sampling_params(self, kwargs: Dict):
        return {
            key: value
            for key, value in kwargs.items()
            if key in ALLOWED_SAMPLING_PARAM_KEYS and value is not None
        }

    def _set_nonblocking(self, fd):
        """Set a file descriptor to non-blocking mode."""
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def _drain_stderr(self):
        """Read and log all remaining stderr output from the subprocess."""
        try:
            self._set_nonblocking(self.process.stderr)
            while True:
                try:
                    line = self.process.stderr.readline()
                    if not line:
                        break
                    logger.error(f"Process stderr (drain): {line.strip()}")
                except (BlockingIOError, IOError):
                    break
        except Exception:
            pass

    def _get_signal_name(self, exit_code):
        """Get human-readable signal name for negative exit codes."""
        if exit_code < 0:
            import signal as sig
            try:
                return f" ({sig.Signals(-exit_code).name})"
            except (ValueError, AttributeError):
                return f" (signal {-exit_code})"
        return ""

    def _check_process_alive(self):
        """Check if the subprocess is still running, raise if it has exited."""
        if self.process.poll() is not None:
            exit_code = self.process.returncode
            # Drain remaining stderr to capture crash details
            self._drain_stderr()
            signal_name = self._get_signal_name(exit_code)
            raise RuntimeError(
                f"Subprocess exited unexpectedly with code {exit_code}{signal_name}"
            )

    def _restart_subprocess(self):
        """Restart the subprocess after a crash."""
        if self._restart_count >= MAX_RESTART_ATTEMPTS:
            raise RuntimeError(
                f"Subprocess has crashed {self._restart_count} times, "
                f"exceeding max restart attempts ({MAX_RESTART_ATTEMPTS}). Giving up."
            )

        self._restart_count += 1
        logger.warning(
            f"Restarting Kimi-Audio subprocess (attempt {self._restart_count}/{MAX_RESTART_ATTEMPTS})..."
        )

        # Clean up old process
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        # Re-launch the subprocess using the same command
        # The _launch_command is saved by the @isolated decorator in isolate.py
        if hasattr(self, '_launch_command'):
            command = self._launch_command
        else:
            # Reconstruct command from command_args
            raise RuntimeError(
                "Cannot restart subprocess: launch command not available. "
                "Please ensure the @isolated decorator saves _launch_command."
            )

        logger.info(f"Restarting with command: {command}")
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
        )
        self._ready = False

        # Wait for the new subprocess to load the model
        self._wait_for_ready()
        logger.info("Subprocess restarted and model loaded successfully.")

    def _wait_for_ready(self):
        """Wait for the subprocess to finish loading the model."""
        if self._ready:
            return
        logger.info("Waiting for Kimi-Audio subprocess model to load...")
        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > MODEL_LOAD_TIMEOUT:
                raise TimeoutError(
                    f"Kimi-Audio model loading timed out after {MODEL_LOAD_TIMEOUT}s"
                )
            self._check_process_alive()
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], READ_POLL_TIMEOUT
            )
            for read in reads:
                if read is self.process.stdout:
                    line = self.process.stdout.readline()
                    if line and "Model loaded" in line:
                        logger.info("Kimi-Audio subprocess model loaded: %s", line.strip())
                        self._ready = True
                        return
                    elif line:
                        logger.debug("Subprocess stdout (loading): %s", line.strip())
                if read is self.process.stderr:
                    err = self.process.stderr.readline()
                    if err:
                        err = err.strip()
                        if any(kw in err for kw in ["INFO", "DEBUG", "Loading", "Building", "loading", "building", "done", "loaded", "%|", "it/s]"]):
                            logger.debug(f"Process stderr (loading): {err}")
                        elif any(kw in err for kw in ["WARNING", "FutureWarning", "UserWarning", "DeprecationWarning", "deprecated", "pkg_resources"]):
                            logger.warning(f"Process stderr (loading): {err}")
                        else:
                            logger.error(f"Process stderr (loading): {err}")

    def _inference(self, prompt: PromptStruct, **kwargs):
        self._wait_for_ready()
        prepared_prompt = self._prepare_prompt(prompt)
        valid_propmt = list(chain(*[self._parse_role_content(item) for item in prepared_prompt]))

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        payload = {
            "messages": valid_propmt,
            "sampling_params": self._collect_sampling_params(kwargs),
            "force_text_output": self._should_force_text_output(prepared_prompt),
        }
        start_time = time.monotonic()

        try:
            return self._do_inference(prefix, payload, start_time)
        except (RuntimeError, TimeoutError) as e:
            should_restart = isinstance(e, TimeoutError) or "exited unexpectedly" in str(e)
            if should_restart:
                logger.error(f"Kimi-Audio subprocess needs restart after inference failure: {e}")
                try:
                    self._restart_subprocess()
                    # Retry inference with new subprocess
                    uid = str(uuid.uuid4())
                    prefix = f"{uid}->"
                    start_time = time.monotonic()
                    return self._do_inference(prefix, payload, start_time)
                except Exception as restart_err:
                    raise RuntimeError(
                        f"Inference failed after subprocess restart: {restart_err}"
                    ) from e
            raise

    def _do_inference(self, prefix, payload, start_time):
        """Execute a single inference request against the subprocess."""
        # Write request to subprocess
        while True:
            self._check_process_alive()
            _, wlist, _ = select.select([], [self.process.stdin], [], WRITE_TIMEOUT)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(payload)}\n")
                self.process.stdin.flush()
                break
            if time.monotonic() - start_time > WRITE_TIMEOUT:
                raise TimeoutError("Timed out waiting to write to subprocess stdin")

        # Read response from subprocess
        while True:
            if time.monotonic() - start_time > INFERENCE_TIMEOUT:
                raise TimeoutError(
                    f"Kimi-Audio inference timed out after {INFERENCE_TIMEOUT}s"
                )
            self._check_process_alive()

            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], READ_POLL_TIMEOUT
            )
            for stream in rlist:
                try:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            res = json.loads(result[len(prefix) :])
                            if len(res) == 1:
                                return res["text"]
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("Kimi-Audio failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            # Classify subprocess stderr by content level
                            if any(kw in err for kw in ["INFO", "DEBUG", "Loading", "Building", "loading", "building", "done", "loaded", "%|", "it/s]"]):
                                logger.debug(f"Process stderr: {err}")
                            elif any(kw in err for kw in ["WARNING", "FutureWarning", "UserWarning", "DeprecationWarning", "deprecated", "pkg_resources"]):
                                logger.warning(f"Process stderr: {err}")
                            else:
                                logger.error(f"Process stderr: {err}")
                except BlockingIOError as e:
                    logger.error(f"BlockingIOError occurred: {str(e)}")
