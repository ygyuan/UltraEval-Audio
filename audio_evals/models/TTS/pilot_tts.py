"""
PilotTTS model wrapper for UltraEval-Audio.

PilotTTS is an LLM-based TTS system from AmapVoice that supports zero-shot
voice cloning, emotion / paralanguage / dialect control. This wrapper runs
the model in an isolated virtual environment (because PilotTTS bundles its
own CosyVoice / Matcha-TTS dependencies) and communicates with it via a
JSON line protocol over stdin/stdout, the same pattern used by VibeVoice
and VoxCPM2.

Reference: https://github.com/xxx/pilot-tts (AmapVoice/PilotTTS)
"""

import json
import logging
import os
import select
import time
import uuid
from typing import Any, Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/PilotTTS/main.py")
class PilotTTS(OfflineModel):
    """
    PilotTTS model wrapper for voice cloning evaluation.

    Args:
        path: Path to the directory that holds PilotTTS pretrained weights
              (``pilot_tts.pt`` / ``pilot_tts_instruct.pt`` /
              ``wav2vec2bert_stats.pt``).
        repo_path: Path to the PilotTTS source repo (the third_party/PilotTTS
              directory cloned from github), needed at runtime so the
              subprocess can ``import pilot_voice`` and find the bundled
              ``cosyvoice`` / ``Matcha-TTS`` packages.
        config_path: PilotTTS YAML config filename. Use
              ``configs/infer_pilot_tts.yaml`` for the base zero-shot model
              and ``configs/infer_pilot_tts_instruct.yaml`` for the instruct
              model. Path is resolved relative to ``repo_path`` if not absolute.
        checkpoint: Checkpoint file name (e.g. ``pilot_tts.pt``). Resolved
              relative to ``path`` if not absolute.
        qwen_path: Path to the Qwen3-0.6B backbone directory.
        w2v_path: Path to the w2v-bert-2.0 feature extractor directory.
        w2v_stats_path: Path to ``wav2vec2bert_stats.pt``.
        cosyvoice_path: Path to the Fun-CosyVoice3 vocoder directory (must
              contain ``campplus.onnx``).
        tokenizer_path: Path to the PilotTTS tokenizer directory.
        language: Default language tag (``zh`` / ``en`` / ``zh-henan`` ...).
        device: Inference device, e.g. ``cuda``.
        sample_params: Optional default sampling parameters (top_p / top_k /
              temperature) that override the YAML config values.
    """

    def __init__(
        self,
        path: str,
        repo_path: str,
        checkpoint: str = "pilot_tts.pt",
        config_path: str = "configs/infer_pilot_tts.yaml",
        qwen_path: str = "",
        w2v_path: str = "",
        w2v_stats_path: str = "",
        cosyvoice_path: str = "",
        tokenizer_path: str = "",
        language: str = "zh",
        device: str = "cuda",
        sample_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isabs(repo_path):
            repo_path = os.path.abspath(repo_path)

        # Resolve checkpoint / config path against the model dir / repo dir
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(path, checkpoint)
        if not os.path.isabs(config_path):
            config_path = os.path.join(repo_path, config_path)

        # Default w2v_stats_path lives next to the checkpoints
        if not w2v_stats_path:
            w2v_stats_path = os.path.join(path, "wav2vec2bert_stats.pt")
        if w2v_stats_path and not os.path.isabs(w2v_stats_path):
            w2v_stats_path = os.path.abspath(w2v_stats_path)

        if qwen_path and not os.path.isabs(qwen_path):
            qwen_path = os.path.abspath(qwen_path)
        if w2v_path and not os.path.isabs(w2v_path):
            w2v_path = os.path.abspath(w2v_path)
        if cosyvoice_path and not os.path.isabs(cosyvoice_path):
            cosyvoice_path = os.path.abspath(cosyvoice_path)
        if tokenizer_path and not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.abspath(tokenizer_path)

        self.command_args = {
            "repo_path": repo_path,
            "config_path": config_path,
            "checkpoint": checkpoint,
            "device": device,
            "language": language,
        }
        if qwen_path:
            self.command_args["qwen_path"] = qwen_path
        if w2v_path:
            self.command_args["w2v_path"] = w2v_path
        if w2v_stats_path:
            self.command_args["w2v_stats_path"] = w2v_stats_path
        if cosyvoice_path:
            self.command_args["cosyvoice_path"] = cosyvoice_path
        if tokenizer_path:
            self.command_args["tokenizer_path"] = tokenizer_path

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Send a synthesis request to the PilotTTS subprocess and wait for
        the resulting audio path.
        """
        text = prompt.get("text")
        prompt_audio = prompt.get("prompt_audio")
        language = prompt.get("language", None)
        emotion = prompt.get("emotion", None)

        if not isinstance(text, str):
            raise TypeError(
                f"Expected 'text' in prompt to be string, but got: {type(text)}"
            )
        if not prompt_audio:
            raise ValueError(
                "PilotTTS requires a 'prompt_audio' (reference wav) in the prompt"
            )

        # subprocess runs in repo dir, so make the prompt_audio absolute
        if not os.path.isabs(prompt_audio):
            prompt_audio = os.path.abspath(prompt_audio)

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        request_data: Dict[str, Any] = {
            "text": text,
            "prompt_audio": prompt_audio,
        }
        if language:
            request_data["language"] = language
        if emotion:
            request_data["emotion"] = emotion

        request = f"{prefix}{json.dumps(request_data, ensure_ascii=False)}\n"
        logger.debug(f"Sending request to PilotTTS process: {request.strip()}")

        # ---- send ----
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 180
            )
            if xlist:
                raise RuntimeError(
                    "PilotTTS stdin broken (select reported error)"
                )
            if not wlist:
                raise TimeoutError("Timeout waiting for PilotTTS stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("PilotTTS process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(f"Error writing to PilotTTS process stdin: {e}")

        # ---- receive ----
        max_wait_time = 600
        start_time = time.time()
        response_line = None

        while time.time() - start_time < max_wait_time:
            try:
                reads, _, xlist = select.select(
                    [self.process.stdout, self.process.stderr],
                    [],
                    [self.process.stdout, self.process.stderr],
                    1.0,
                )
                if xlist:
                    raise RuntimeError(
                        "PilotTTS stdout/stderr broken (select reported error)"
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            if any(
                                kw in err
                                for kw in [
                                    "INFO",
                                    "DEBUG",
                                    "Loading",
                                    "loading",
                                    "loaded",
                                    "%|",
                                    "it/s]",
                                    # CosyVoice3 / shell startup chatter that
                                    # the subprocess redirects to stderr.
                                    "bashrc",
                                    "Now using node",
                                    "streaming",
                                    "ttsfrd",
                                    "wetext",
                                    "PilotTTS loaded",
                                    "ENABLE_RTF",
                                    "RTF:",
                                ]
                            ):
                                logger.debug(f"PilotTTS stderr: {err}")
                            elif any(
                                kw in err
                                for kw in [
                                    "WARNING",
                                    "FutureWarning",
                                    "UserWarning",
                                    "DeprecationWarning",
                                    "deprecated",
                                    "not found close signal",
                                ]
                            ):
                                logger.warning(f"PilotTTS stderr: {err}")
                            else:
                                logger.error(f"PilotTTS stderr: {err}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            response_line = result[len(prefix):]
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            return response_line
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"PilotTTS failed: {result}")
                        else:
                            # Subprocess stdout is reserved for protocol
                            # responses; anything else is unexpected stray
                            # output and is logged at debug level only.
                            logger.debug(f"PilotTTS stdout (ignored): {result}")
            except Exception as e:
                if "PilotTTS" in str(e):
                    raise
                raise RuntimeError(f"Error reading from PilotTTS process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from PilotTTS process for request {uid}"
            )
