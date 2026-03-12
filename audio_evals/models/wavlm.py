import logging
import os
import sys
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import gdown
from audio_evals.constants import DEFAULT_MODEL_PATH


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/simo/simo.py")
class WavLM(OfflineModel):
    def __init__(
        self,
        path: str = "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view",
        sample_params: Dict = None,
    ):
        if path.startswith("https://drive.google.com"):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    @staticmethod
    def _download_model(url: str) -> str:
        """Download model from Google Drive if not exists locally.

        Args:
            url: Google Drive share URL

        Returns:
            str: Local path where model is downloaded
        """
        try:
            logger.info(
                f"Downloading model from Google Drive: {url}, need use proxy to access Google Drive if in China."
            )
            # Extract file ID from URL
            file_id = url.split("/")[-2]
            output_dir = os.path.join(DEFAULT_MODEL_PATH, "wavlm")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "wavlm_large_finetune.pth")
            if os.path.exists(output_path):
                logger.info(f"Model already present locally, skip download: {output_path}")
                return output_path

            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=output_path,
                quiet=False,
            )
            logger.info(f"Model downloaded to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        audio_paths = prompt["audios"]
        assert len(audio_paths) == 2, "wav lm must be used with two audio files."
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # 检查进程状态，如果进程已终止则重新启动
        if self.process.poll() is not None:
            logger.error("Subprocess has terminated unexpectedly")
            # 尝试读取子进程的错误输出以获取更多信息
            try:
                stdout, stderr = self.process.communicate(timeout=5)
                if stderr:
                    logger.error(f"Subprocess stderr: {stderr}")
                if stdout:
                    logger.info(f"Subprocess stdout: {stdout}")
            except Exception as e:
                logger.warning(f"Failed to read subprocess output: {e}")
            
            raise RuntimeError("Subprocess has terminated unexpectedly. This may be due to GPU memory issues, model loading failures, or other runtime errors.")

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                try:
                    self.process.stdin.write(f"{prefix}{','.join(audio_paths)}\n")
                    self.process.stdin.flush()
                    logger.debug("wavlm prompt written to stdin")
                    break
                except BrokenPipeError:
                    logger.error("BrokenPipeError: Subprocess terminated during write operation")
                    raise RuntimeError("Subprocess terminated during communication. Check GPU memory and model availability.")
                except Exception as e:
                    logger.error(f"Error writing to subprocess: {e}")
                    raise

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            try:
                                self.process.stdin.write("{}close\n".format(prefix))
                                self.process.stdin.flush()
                            except BrokenPipeError:
                                logger.warning("BrokenPipeError when sending close signal, but result already received")
                            return float(result[len(prefix) :])
                        elif result.startswith("Error:"):
                            raise RuntimeError("wav lm failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            # Classify subprocess stderr by content level
                            if any(kw in err for kw in ["INFO:", "DEBUG:"]):
                                logger.debug(f"Process stderr: {err}")
                            elif any(kw in err for kw in ["WARNING:", "FutureWarning", "UserWarning", "DeprecationWarning"]):
                                logger.warning(f"Process stderr: {err}")
                            else:
                                logger.error(f"Process stderr: {err}")
                # logger.info("Waiting for wav lm result...")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
