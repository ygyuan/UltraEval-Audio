import json
import logging
import os
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/qwen3-omni/main.py",
)
class Qwen3Omni(OfflineModel):
    def __init__(
        self,
        path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        speech: bool = False,
        speaker: str = "Ethan",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "Qwen/Qwen3-Omni-30B-A3B-Instruct" and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "speaker": speaker,
        }
        if speech:
            self.command_args["speech"] = ""

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

    def _inference(self, prompt: PromptStruct, **kwargs):
        conversation = [self._parse_role_content(item) for item in prompt]
        import uuid
        import time

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Check if subprocess is still alive before writing
        if self.process.poll() is not None:
            raise RuntimeError(
                f"qwen3-omni subprocess has exited with code {self.process.returncode}"
            )

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(conversation)}\n")
                self.process.stdin.flush()
                print("already write in")
                break

        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()
        while True:
            # Check if subprocess is still alive
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"qwen3-omni subprocess has exited with code {self.process.returncode} "
                    f"while waiting for response"
                )

            # Check for overall timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise RuntimeError(
                    f"qwen3-omni inference timed out after {max_wait_time}s"
                )

            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if result:
                        if result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            res = json.loads(result[len(prefix) :])
                            res["text"] = res["text"].split("assistant")[-1].strip()
                            logger.info("return output: %s", json.dumps(res, ensure_ascii=False))
                            return json.dumps(res, ensure_ascii=False)
                        elif result.startswith("Error:"):
                            raise RuntimeError("qwen3-omni failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
