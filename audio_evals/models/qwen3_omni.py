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
    "audio_evals/lib/qwen3-omni/main.py",
)
class Qwen3Omni(OfflineModel):
    def __init__(
        self,
        path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        speech: bool = False,
        speaker: str = "Ethan",
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 512,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "Qwen/Qwen3-Omni-30B-A3B-Instruct" and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "speaker": speaker,
            "thinker_max_new_tokens": str(thinker_max_new_tokens),
            "talker_max_new_tokens": str(talker_max_new_tokens),
        }
        if speech:
            self.command_args["speech"] = ""

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
        logger.info("Waiting for subprocess model to load...")
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
                        logger.info("Subprocess model loaded: %s", line.strip())
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
                        # Clean up text: remove leading role marker if present
                        text = res.get("text", "")
                        if "\nassistant\n" in text:
                            text = text.split("\nassistant\n", 1)[-1].strip()
                        res["text"] = text
                        if "audio" not in res:
                            return res["text"]
                        return json.dumps(res, ensure_ascii=False)
                    elif result.startswith("Error:"):
                        raise RuntimeError(f"qwen3-omni failed: {result.strip()}")
                    else:
                        logger.debug("Subprocess stdout: %s", result.strip())
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        logger.debug("Subprocess stderr: %s", error_output.strip())