"""Fun-Audio-Chat model integration.

The actual inference runs in an isolated Python subprocess launched by
``audio_evals.isolate.isolated`` because Fun-Audio-Chat has a very
specific dependency stack (transformers 4.52.3 + Python 3.12 + CosyVoice3).
This wrapper only handles IPC: sending request JSON to the subprocess and
parsing the response.
"""

import json
import logging
import os
import select
import threading
import time
import uuid
from typing import Any, Dict

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

# Timeout constants (seconds). Fun-Audio-Chat-8B is a fairly large model and
# can take a while to load, especially with the CosyVoice3 detokenizer.
WRITE_TIMEOUT = 60
READ_POLL_TIMEOUT = 1.0
INFERENCE_TIMEOUT = 900
MODEL_LOAD_TIMEOUT = 1800
HEARTBEAT_INTERVAL = 30.0


@isolated(
    "audio_evals/lib/funaudio_chat/main.py",
)
class FunAudioChat(OfflineModel):
    """Isolated wrapper around the official Fun-Audio-Chat inference script."""

    def __init__(
        self,
        model_path: str = "FunAudioLLM/Fun-Audio-Chat-8B",
        tts_model_path: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        repo_root: str = None,
        tts_out_dir: str = None,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        # Resolve the main model checkpoint (accepts local path or HF repo id).
        if not os.path.exists(model_path):
            model_path = self._resolve_model_path(model_path)

        # Resolve the CosyVoice3 detokenizer checkpoint.
        if not os.path.exists(tts_model_path):
            tts_model_path = self._resolve_model_path(tts_model_path)

        # Resolve the Fun-Audio-Chat source repo (needed for utils / funaudiochat
        # imports inside the subprocess).
        if repo_root is None:
            repo_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "init_model",
                    "FunAudioLLM", "Fun-Audio-Chat",
                )
            )
        if not os.path.isdir(repo_root):
            logger.warning(
                "Fun-Audio-Chat repo not found at %s; subprocess imports "
                "(funaudiochat / utils) may fail.", repo_root,
            )

        self.command_args = {
            "model_path": model_path,
            "tts_model_path": tts_model_path,
            "repo_root": repo_root,
        }
        if tts_out_dir:
            self.command_args["tts_out_dir"] = tts_out_dir

        self._ready = False
        self._stderr_thread = None
        self._stderr_stop = None
        super().__init__(is_chat=True, sample_params=sample_params)
        # NOTE: We intentionally do NOT call ``_wait_for_ready`` here.
        # ``@isolated`` launches the subprocess in a wrapper that runs
        # *after* ``original_init`` returns, so ``self.process`` does not
        # yet exist at this point. The actual wait happens lazily on the
        # first ``_inference`` call.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_path(path_or_repo: str) -> str:
        """Accept either a local directory or a HF/ModelScope repo id."""
        if os.path.exists(path_or_repo):
            return path_or_repo
        try:
            return FunAudioChat._download_model(path_or_repo)
        except Exception as err:
            logger.warning(
                "Failed to download %s via HuggingFace (%s); trying ModelScope",
                path_or_repo, err,
            )
            return FunAudioChat._download_model_from_modelscope(path_or_repo)

    def _check_process_alive(self):
        if self.process.poll() is not None:
            exit_code = self.process.returncode
            raise RuntimeError(
                f"Fun-Audio-Chat subprocess exited unexpectedly with code {exit_code}"
            )

    def _start_stderr_drain(self):
        """Continuously drain the subprocess stderr.

        Needed because transformers / CosyVoice3 / tqdm produce a lot of
        output during model loading. If we do not actively read stderr the
        OS pipe buffer (~64KB) fills up and the subprocess *blocks on
        print*, which prevents it from ever emitting the ``Model loaded``
        marker and deadlocks ``_wait_for_ready``.
        """
        if self._stderr_thread is not None:
            return
        self._stderr_stop = threading.Event()

        def _drain():
            stderr = self.process.stderr
            while not self._stderr_stop.is_set():
                try:
                    line = stderr.readline()
                except Exception:
                    break
                if not line:
                    # EOF: subprocess exited.
                    break
                stripped = line.rstrip()
                if stripped:
                    logger.debug("[funaudio_chat stderr] %s", stripped)

        self._stderr_thread = threading.Thread(
            target=_drain,
            name="funaudio-chat-stderr-drain",
            daemon=True,
        )
        self._stderr_thread.start()

    def _wait_for_ready(self):
        if self._ready:
            return
        # Make sure the stderr drainer is running before we block on stdout,
        # otherwise a full stderr pipe will freeze the child process.
        self._start_stderr_drain()

        logger.info(
            "Waiting for Fun-Audio-Chat subprocess model to load "
            "(pid=%d, timeout=%ds)...",
            self.process.pid, MODEL_LOAD_TIMEOUT,
        )
        start_time = time.monotonic()
        last_heartbeat = start_time
        buffer = ""

        while True:
            now = time.monotonic()
            if now - start_time > MODEL_LOAD_TIMEOUT:
                raise TimeoutError(
                    f"Fun-Audio-Chat model loading timed out after "
                    f"{MODEL_LOAD_TIMEOUT}s (pid={self.process.pid})"
                )
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                logger.info(
                    "Fun-Audio-Chat subprocess still loading "
                    "(pid=%d, elapsed=%.0fs)...",
                    self.process.pid, now - start_time,
                )
                last_heartbeat = now

            self._check_process_alive()
            reads, _, _ = select.select(
                [self.process.stdout], [], [], READ_POLL_TIMEOUT
            )
            if not reads:
                continue

            # Read whatever is currently available (non-line-based) so that
            # tqdm-style '\r' progress output does not block readline().
            try:
                chunk = os.read(self.process.stdout.fileno(), 4096)
            except OSError:
                continue
            if not chunk:
                # EOF on stdout: the subprocess is going away.
                self._check_process_alive()
                continue

            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            buffer += text

            # Split on both '\n' and '\r' so carriage-return progress bars
            # don't accumulate forever.
            while True:
                # Find the earliest line terminator.
                idx_n = buffer.find("\n")
                idx_r = buffer.find("\r")
                if idx_n == -1 and idx_r == -1:
                    break
                if idx_n == -1:
                    idx = idx_r
                elif idx_r == -1:
                    idx = idx_n
                else:
                    idx = min(idx_n, idx_r)
                line = buffer[:idx]
                buffer = buffer[idx + 1:]
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if "Model loaded" in line_stripped:
                    logger.info(
                        "Fun-Audio-Chat subprocess model loaded "
                        "(pid=%d, elapsed=%.1fs): %s",
                        self.process.pid,
                        time.monotonic() - start_time,
                        line_stripped,
                    )
                    self._ready = True
                    return
                logger.info(
                    "[funaudio_chat stdout] %s", line_stripped
                )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _inference(self, prompt: PromptStruct, **kwargs: Any) -> str:
        self._wait_for_ready()

        # The subprocess expects {"prompt": [...], "task": ...}.
        payload: Dict[str, Any] = {"prompt": prompt}
        if kwargs:
            # Only forward keys the subprocess understands. Any unknown kwargs
            # (e.g. generation sampling params) are simply ignored.
            for k in ("task",):
                if k in kwargs and kwargs[k] is not None:
                    payload[k] = kwargs[k]

        uid = uuid.uuid4().hex
        prefix = f"{uid}->"
        request_line = prefix + json.dumps(payload, ensure_ascii=False) + "\n"
        start_time = time.monotonic()

        # Write the request to stdin.
        while True:
            self._check_process_alive()
            _, wlist, _ = select.select([], [self.process.stdin], [], WRITE_TIMEOUT)
            if wlist:
                self.process.stdin.write(request_line)
                self.process.stdin.flush()
                logger.debug("Fun-Audio-Chat: request written (uid=%s)", uid)
                break
            if time.monotonic() - start_time > WRITE_TIMEOUT:
                raise TimeoutError(
                    "Timed out writing to Fun-Audio-Chat subprocess stdin"
                )

        # Read the response.
        while True:
            if time.monotonic() - start_time > INFERENCE_TIMEOUT:
                raise TimeoutError(
                    f"Fun-Audio-Chat inference timed out after {INFERENCE_TIMEOUT}s"
                )
            self._check_process_alive()
            reads, _, _ = select.select(
                [self.process.stdout], [], [], READ_POLL_TIMEOUT
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if not result:
                        continue
                    if result.startswith(prefix):
                        # Close the round-trip.
                        try:
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                        except BrokenPipeError:
                            pass
                        try:
                            res = json.loads(result[len(prefix):])
                        except json.JSONDecodeError as err:
                            raise RuntimeError(
                                f"Fun-Audio-Chat returned non-JSON: {result!r} ({err})"
                            )
                        logger.info("Fun-Audio-Chat output: %s", res)
                        if "audio" in res and res["audio"]:
                            # Spoken-QA / S2S response: caller expects JSON so that
                            # ``extract_audio`` post-processing can pick up the wav.
                            return json.dumps(res, ensure_ascii=False)
                        return res.get("text", "") or ""
                    if result.startswith("Error:"):
                        raise RuntimeError(
                            f"Fun-Audio-Chat failed: {result.strip()}"
                        )
                    logger.debug("Subprocess stdout: %s", result.strip())
