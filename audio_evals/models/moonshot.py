from itertools import chain
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
READ_POLL_TIMEOUT = 5.0
# Single inference upper bound. asc-moan / speech tasks can be slow on the
# first sample because the subprocess lazily JIT-compiles flash-attn kernels
# and warms up CUDA caches; 1800s is a comfortable ceiling.
INFERENCE_TIMEOUT = 1800
# When the subprocess is still loading the model, the very first inference
# call has to wait for: weight loading (NFS, ~30GB) + flash-attn JIT +
# detokenizer init. Allow up to MODEL_LOAD_TIMEOUT *in addition to* the
# normal INFERENCE_TIMEOUT for that first call.
MODEL_LOAD_TIMEOUT = 1800
# Max number of subprocess restart attempts after a crash.
MAX_RESTART_ATTEMPTS = 3
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


def _classify_stderr_level(line: str) -> int:
    """Classify a stderr line to a logging level for forwarding."""
    if any(
        kw in line
        for kw in [
            "INFO", "DEBUG", "Loading", "Building", "loading", "building",
            "done", "loaded", "%|", "it/s]",
        ]
    ):
        return logging.DEBUG
    if any(
        kw in line
        for kw in [
            "WARNING", "FutureWarning", "UserWarning", "DeprecationWarning",
            "deprecated", "pkg_resources",
        ]
    ):
        return logging.WARNING
    return logging.ERROR


@isolated("audio_evals/lib/Kimi-Audio/main.py")
class KimiAudioModel(OfflineModel):
    def __init__(
        self,
        model_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        speech: bool = False,
        lazy_detokenizer: bool = False,
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
        # Skip detokenizer loading at startup. Saves tens of seconds per
        # worker; safe whenever the task only needs text outputs (ASR /
        # S2TT / classification / emotion / scene).
        if lazy_detokenizer and not speech:
            self.command_args["lazy_detokenizer"] = ""

        self.speech = speech
        self._restart_count = 0
        super().__init__(is_chat=True, sample_params=sample_params)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _parse_role_content(self, role_content: Dict):
        assert isinstance(
            role_content["contents"], list
        ), "prompt should be list not string"

        res = []
        for c in role_content["contents"]:
            res.append(
                {
                    "role": role_content["role"],
                    "message_type": c["type"],
                    "content": c["value"],
                }
            )
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
            for content in item.get("contents", []):
                if content.get("type") == "audio":
                    has_audio = True
                elif content.get("type") == "text":
                    text = str(content.get("value", "")).strip().lower()
                    if any(
                        keyword in text
                        for keyword in [
                            "speak", "speech", "voice", "read aloud",
                            "audio reply", "say it",
                        ]
                    ):
                        return False
        return has_audio

    def _collect_sampling_params(self, kwargs: Dict):
        return {
            key: value
            for key, value in kwargs.items()
            if key in ALLOWED_SAMPLING_PARAM_KEYS and value is not None
        }

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------
    def _get_signal_name(self, exit_code):
        if exit_code < 0:
            import signal as sig
            try:
                return f" ({sig.Signals(-exit_code).name})"
            except (ValueError, AttributeError):
                return f" (signal {-exit_code})"
        return ""

    def _check_process_alive(self):
        if self.process.poll() is not None:
            exit_code = self.process.returncode
            signal_name = self._get_signal_name(exit_code)
            raise RuntimeError(
                f"Subprocess exited unexpectedly with code {exit_code}{signal_name}"
            )

    def _restart_subprocess(self):
        if self._restart_count >= MAX_RESTART_ATTEMPTS:
            raise RuntimeError(
                f"Subprocess has crashed {self._restart_count} times, "
                f"exceeding max restart attempts ({MAX_RESTART_ATTEMPTS}). Giving up."
            )

        self._restart_count += 1
        logger.warning(
            f"Restarting Kimi-Audio subprocess (attempt {self._restart_count}/{MAX_RESTART_ATTEMPTS})..."
        )

        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        # ``_launch_command`` is saved by the @isolated decorator.
        if not hasattr(self, "_launch_command"):
            raise RuntimeError(
                "Cannot restart subprocess: launch command not available. "
                "Please ensure the @isolated decorator saves _launch_command."
            )

        logger.info(f"Restarting with command: {self._launch_command}")
        self.process = subprocess.Popen(
            self._launch_command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
        )
        logger.info("Subprocess restarted; will block on the next inference call until the model is ready.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _inference(self, prompt: PromptStruct, **kwargs):
        prepared_prompt = self._prepare_prompt(prompt)
        valid_propmt = list(
            chain(*[self._parse_role_content(item) for item in prepared_prompt])
        )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        payload = {
            "messages": valid_propmt,
            "sampling_params": self._collect_sampling_params(kwargs),
            "force_text_output": self._should_force_text_output(prepared_prompt),
        }
        # The very first call has to wait for model loading on top of the
        # actual inference, so we give it the full load budget plus inference
        # budget. Subsequent calls use just INFERENCE_TIMEOUT.
        deadline = time.monotonic() + (
            INFERENCE_TIMEOUT
            if getattr(self, "_first_inference_done", False)
            else MODEL_LOAD_TIMEOUT + INFERENCE_TIMEOUT
        )

        try:
            result = self._do_inference(prefix, payload, deadline)
            self._first_inference_done = True
            return result
        except (RuntimeError, TimeoutError) as e:
            should_restart = isinstance(e, TimeoutError) or "exited unexpectedly" in str(e)
            if not should_restart:
                raise
            logger.error(
                f"Kimi-Audio subprocess needs restart after inference failure: {e}"
            )
            try:
                self._restart_subprocess()
            except Exception as restart_err:
                raise RuntimeError(
                    f"Inference failed and subprocess restart also failed: {restart_err}"
                ) from e
            # Retry once with the fresh subprocess. Reset the "first
            # inference" flag so we again allow the long load budget.
            self._first_inference_done = False
            uid = str(uuid.uuid4())
            prefix = f"{uid}->"
            deadline = time.monotonic() + MODEL_LOAD_TIMEOUT + INFERENCE_TIMEOUT
            try:
                result = self._do_inference(prefix, payload, deadline)
                self._first_inference_done = True
                return result
            except Exception as retry_err:
                raise RuntimeError(
                    f"Inference failed after subprocess restart: {retry_err}"
                ) from e

    def _do_inference(self, prefix, payload, deadline):
        """Send one request to the subprocess and read its response.

        We deliberately keep the IO model identical to the original simple
        version: a single ``select`` over both ``stdout`` and ``stderr``
        with blocking ``readline()``. We do **not** put the pipes in
        non-blocking mode and do **not** spawn a background drain thread,
        because either of those have proven to cause the subprocess's
        ``logging``/``print`` calls to silently stall on first use.
        """
        # ---- Send request ----
        write_deadline = min(deadline, time.monotonic() + WRITE_TIMEOUT)
        while True:
            self._check_process_alive()
            remaining = write_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting to write to subprocess stdin")
            _, wlist, _ = select.select([], [self.process.stdin], [], min(remaining, 1.0))
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(payload)}\n")
                self.process.stdin.flush()
                break

        # ---- Read response ----
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Kimi-Audio inference timed out after {INFERENCE_TIMEOUT}s "
                    f"(or {MODEL_LOAD_TIMEOUT + INFERENCE_TIMEOUT}s on the first call)"
                )
            self._check_process_alive()

            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr],
                [],
                [],
                min(remaining, READ_POLL_TIMEOUT),
            )
            if not rlist:
                continue

            for stream in rlist:
                if stream is self.process.stdout:
                    line = self.process.stdout.readline()
                    if not line:
                        # EOF on stdout — process exited; let the next loop
                        # iteration surface the real error via _check_process_alive.
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(prefix):
                        # Tell the subprocess we received the result so it
                        # stops re-emitting the close-handshake retries.
                        try:
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                        except Exception as close_err:
                            logger.warning(
                                f"Failed to send close handshake: {close_err}"
                            )
                        body = line[len(prefix):]
                        try:
                            res = json.loads(body)
                        except json.JSONDecodeError:
                            return body
                        if isinstance(res, dict) and len(res) == 1 and "text" in res:
                            return res["text"]
                        return body
                    if line.startswith("Error:"):
                        raise RuntimeError(f"Kimi-Audio failed: {line}")
                    # Anything else is loading progress / informational
                    # output from the subprocess — log it so users can see
                    # that loading is making progress.
                    logger.info("Subprocess stdout: %s", line)
                elif stream is self.process.stderr:
                    err = self.process.stderr.readline()
                    if not err:
                        continue
                    err = err.strip()
                    if err:
                        logger.log(_classify_stderr_level(err), "Process stderr: %s", err)
