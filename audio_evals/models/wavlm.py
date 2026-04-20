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
    # Class-level lock to protect _io_lock initialization (double-checked locking)
    _init_lock = __import__('threading').Lock()

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
        import time
        import threading

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Ensure subprocess is alive before attempting I/O; auto-restart if dead
        self.ensure_process_alive()

        # Thread-safe lazy initialization of _io_lock using double-checked locking
        if not hasattr(self, '_io_lock'):
            with WavLM._init_lock:
                if not hasattr(self, '_io_lock'):
                    self._io_lock = threading.Lock()

        with self._io_lock:
            # Write request
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

            # Read response with timeout (simo.py retries 3 times with 3s each = ~12s max)
            max_wait_time = 120  # generous timeout for slow GPU inference
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
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
                except BlockingIOError as e:
                    logger.error(f"BlockingIOError occurred: {str(e)}")
                    continue

            # Timeout reached
            raise TimeoutError(
                f"Timeout ({max_wait_time}s) waiting for WavLM simo result for request {uid}"
            )
