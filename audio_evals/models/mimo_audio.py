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
        """Check if the subprocess is still running, raise if it has exited."""
        if self.process.poll() is not None:
            exit_code = self.process.returncode
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
