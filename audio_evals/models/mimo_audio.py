import json
import logging
import os
import select
import time
import uuid
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
WRITE_TIMEOUT = 60
READ_POLL_TIMEOUT = 1.0
INFERENCE_TIMEOUT = 900  # Max total time for a single inference call
MODEL_LOAD_TIMEOUT = 600  # Max time to wait for model loading


@isolated(
    "audio_evals/lib/MiMo-Audio/main.py",
)
class MiMoAudio(OfflineModel):
    def __init__(
        self,
        model_path: str = "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        tokenizer_path: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(model_path):
            model_path = self._download_model(model_path)
        if not os.path.exists(tokenizer_path):
            tokenizer_path = self._download_model(tokenizer_path)

        self.command_args = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
        }

        self._ready = False
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_content(self, content: Dict):
        assert "type" in content
        return {"type": content["type"], content["type"]: content["value"]}

    def _parse_role_content(self, role_content: Dict):
        for k in ["contents"]:
            if isinstance(role_content[k], list):
                role_content["content"] = [
                    self._parse_content(item) for item in role_content.pop(k)
                ]
            else:
                role_content["content"] = role_content.pop(k)
        return role_content

    def _check_process_alive(self):
        """Check if the subprocess is still running.

        If the subprocess has terminated unexpectedly, try to restart it via
        the ``ensure_process_alive`` hook installed by ``@isolated``. When the
        restart succeeds we reset ``self._ready`` so that the caller will
        re-enter ``_wait_for_ready`` and wait for the new subprocess to
        finish loading the model. When the restart hook is unavailable or
        also fails, propagate ``RuntimeError`` so the outer evaluation loop
        can record the failure.
        """
        if self.process.poll() is None:
            return

        exit_code = self.process.returncode
        # Prefer the auto-restart helper provided by @isolated. It also
        # enforces a hard cap on consecutive restarts so a persistently
        # broken worker eventually surfaces as an error.
        ensure_alive = getattr(self, "ensure_process_alive", None)
        if callable(ensure_alive):
            try:
                ensure_alive()
                # Newly-spawned subprocess still needs to load the model.
                self._ready = False
                logger.warning(
                    "MiMo-Audio subprocess restarted (previous exit code: %d); "
                    "will wait for the new instance to load the model.",
                    exit_code,
                )
                return
            except Exception as restart_err:
                raise RuntimeError(
                    f"Subprocess exited unexpectedly with code {exit_code} "
                    f"and auto-restart failed: {restart_err}"
                ) from restart_err

        raise RuntimeError(
            f"Subprocess exited unexpectedly with code {exit_code}"
        )

    def _wait_for_ready(self):
        """Wait for the subprocess to finish loading the model."""
        if self._ready:
            return
        logger.info("Waiting for MiMo-Audio subprocess model to load...")
        start_time = time.monotonic()
        while True:
            if time.monotonic() - start_time > MODEL_LOAD_TIMEOUT:
                raise TimeoutError(
                    f"Model loading timed out after {MODEL_LOAD_TIMEOUT}s"
                )
            self._check_process_alive()
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], READ_POLL_TIMEOUT
            )
            for read in reads:
                if read is self.process.stdout:
                    line = self.process.stdout.readline()
                    if line and "Model loaded" in line:
                        logger.info("MiMo-Audio subprocess model loaded: %s", line.strip())
                        self._ready = True
                        return
                    elif line:
                        logger.debug("Subprocess stdout (loading): %s", line.strip())
                if read is self.process.stderr:
                    err = self.process.stderr.readline()
                    if err:
                        logger.debug("Subprocess stderr (loading): %s", err.strip())

    def _inference(self, prompt: PromptStruct, **kwargs):
        try:
            return self._do_inference(prompt)
        except (RuntimeError, TimeoutError) as e:
            # Try to recover from subprocess crash / hang by forcibly
            # restarting the worker once and retrying the same request.
            # This protects long-running evaluations (e.g. asc-moan with
            # ~74k samples) from hanging forever when a single sample
            # crashes the subprocess.
            should_restart = (
                isinstance(e, TimeoutError)
                or "exited unexpectedly" in str(e)
                or "auto-restart failed" in str(e)
            )
            restart_fn = getattr(self, "restart_process", None)
            if not (should_restart and callable(restart_fn)):
                raise

            logger.error(
                "MiMo-Audio subprocess needs forced restart after inference "
                "failure: %s",
                e,
            )
            try:
                # Kill the (possibly hung) subprocess and spawn a fresh one.
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=10)
                    except Exception:
                        try:
                            self.process.kill()
                        except Exception:
                            pass
                restart_fn()
                self._ready = False
                return self._do_inference(prompt)
            except Exception as restart_err:
                raise RuntimeError(
                    f"Inference failed after subprocess restart: {restart_err}"
                ) from e

    def _do_inference(self, prompt: PromptStruct):
        self._wait_for_ready()
        conversation = [self._parse_role_content(item) for item in prompt]

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        start_time = time.monotonic()

        # Write request to subprocess
        while True:
            self._check_process_alive()
            _, wlist, _ = select.select([], [self.process.stdin], [], WRITE_TIMEOUT)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(conversation)}\n")
                self.process.stdin.flush()
                logger.debug("Request written to subprocess")
                break
            if time.monotonic() - start_time > WRITE_TIMEOUT:
                raise TimeoutError("Timed out waiting to write to subprocess stdin")

        # Read response from subprocess
        while True:
            if time.monotonic() - start_time > INFERENCE_TIMEOUT:
                raise TimeoutError(
                    f"Inference timed out after {INFERENCE_TIMEOUT}s"
                )
            self._check_process_alive()

            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], READ_POLL_TIMEOUT
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if not result:
                        continue
                    if result.startswith(prefix):
                        # Send close signal to subprocess
                        self.process.stdin.write(f"{prefix}close\n")
                        self.process.stdin.flush()
                        res = json.loads(result[len(prefix):])
                        logger.info("Subprocess returned output: %s", res)
                        text = res.get("text", "")
                        # Clean up text: remove leading role marker if present
                        if "\nassistant\n" in text:
                            text = text.split("\nassistant\n", 1)[-1].strip()
                        return text
                    elif result.startswith("Error:"):
                        raise RuntimeError(f"MiMo-Audio failed: {result.strip()}")
                    else:
                        logger.debug("Subprocess stdout: %s", result.strip())
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        logger.debug("Subprocess stderr: %s", error_output.strip())
