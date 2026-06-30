"""Subprocess entry point for NVIDIA Canary-Qwen-2.5B (SALM) ASR model.

This file is launched by ``audio_evals.models.asr.canary_qwen.CanaryQwen``
(via the ``@isolated`` decorator) inside an isolated virtualenv that has
NeMo + transformers installed.

It speaks the same line-based JSON protocol as the other ASR wrappers:

    request   :  ``<uuid>-> {"audio": "/path/to.wav", "kwargs": {...}}\n``
    response  :  ``<uuid>-> {"content": "...", "raw_text": "..."}\n``
    close     :  ``<uuid>-> close\n``

Canary-Qwen-2.5B is a Speech-Augmented Language Model (SALM) that combines
a Canary speech encoder with a Qwen LLM decoder.  The official way to
transcribe with it is::

    from nemo.collections.speechlm2.models import SALM
    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    answer_ids = model.generate(
        prompts=[[
            {"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}
        ]],
        audios=[audio_path],
    )
    text = model.tokenizer.ids_to_text(answer_ids[0].cpu())

Reference:
    https://huggingface.co/nvidia/canary-qwen-2.5b
"""

import argparse
import json
import logging
import os
import re
import select
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("NEMO_LOGGING_LEVEL", "WARNING")

import torch  # noqa: E402


logger = logging.getLogger("canary-qwen")
logging.basicConfig(
    level=os.environ.get("CANARY_QWEN_LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


DEFAULT_ASR_PROMPT = "Transcribe the following:"


_LEADING_ROLE_RE = re.compile(
    r"^\s*(?:<\|?(?:assistant|system|user)\|?>?|assistant|system|user)\s*[:\n]?\s*",
    re.IGNORECASE,
)


# Candidate local directories for the LLM that SALM expects in
# ``pretrained_llm`` (default: ``Qwen/Qwen3-1.7B``).  We probe these in
# order; the first one that contains ``config.json`` wins.
_DEFAULT_LOCAL_LLM_CANDIDATES = (
    "init_model/Qwen3-1.7B",
    "init_model/Qwen/Qwen3-1.7B",
    "init_model/qwen/Qwen3-1.7B",
)


def _clean_text(text):
    if not isinstance(text, str):
        return "" if text is None else str(text)
    return _LEADING_ROLE_RE.sub("", text).strip()


def _resolve_local_llm_dir():
    """Return absolute path of a local Qwen3-1.7B checkpoint, or ``None``.

    The lookup is rooted at the current working directory (which is the
    UltraEval-Audio project root when launched via the standard wrapper).
    Allows override via the ``CANARY_QWEN_LOCAL_LLM`` environment variable.
    """
    override = os.environ.get("CANARY_QWEN_LOCAL_LLM", "").strip()
    candidates = []
    if override:
        candidates.append(override)
    candidates.extend(_DEFAULT_LOCAL_LLM_CANDIDATES)

    for cand in candidates:
        if not cand:
            continue
        abs_cand = cand if os.path.isabs(cand) else os.path.abspath(cand)
        cfg_file = os.path.join(abs_cand, "config.json")
        if os.path.isdir(abs_cand) and os.path.isfile(cfg_file):
            return abs_cand
    return None


def _patch_pretrained_llm_to_local(ckpt_dir):
    """Rewrite ``<ckpt_dir>/config.json::pretrained_llm`` to a local path.

    SALM (``nemo.collections.speechlm2.models.SALM``) instantiates an
    ``AutoTokenizer`` / loads HF weights from whatever string is stored in
    the ``pretrained_llm`` field of the checkpoint config.  In offline /
    air-gapped environments the default value (``"Qwen/Qwen3-1.7B"``) makes
    that call fail with ``LocalEntryNotFoundError``.

    This helper is idempotent:
      * If ``pretrained_llm`` already points at an existing local
        directory, nothing is changed.
      * Otherwise, if a local Qwen3-1.7B checkpoint can be located on disk
        (see ``_resolve_local_llm_dir``), the field is rewritten in-place
        and a one-line backup of the original value is kept under
        ``pretrained_llm_original`` for traceability.
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        logger.warning(
            "No config.json found in %s; skipping pretrained_llm patch.",
            ckpt_dir,
        )
        return

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read %s for patching: %s", cfg_path, e)
        return

    current = cfg.get("pretrained_llm", "")
    if isinstance(current, str) and current:
        # Already a local directory that exists -> nothing to do.
        if os.path.isdir(current):
            logger.info(
                "pretrained_llm already points to a local directory: %s",
                current,
            )
            return

    local_llm = _resolve_local_llm_dir()
    if not local_llm:
        logger.warning(
            "pretrained_llm=%r is not a local path and no local Qwen3-1.7B "
            "checkpoint was found under init_model/.  SALM will likely "
            "fail to load in offline mode.  Set CANARY_QWEN_LOCAL_LLM to "
            "override.",
            current,
        )
        return

    if cfg.get("pretrained_llm") == local_llm:
        return  # Already patched.

    cfg.setdefault("pretrained_llm_original", current)
    cfg["pretrained_llm"] = local_llm
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info(
            "Patched pretrained_llm in %s: %r -> %r",
            cfg_path, current, local_llm,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to write patched %s: %s", cfg_path, e)


def _load_canary_qwen(path, device, dtype):
    """Load the Canary-Qwen SALM model.

    Args:
        path: local directory of the checkpoint or a HF repo id.
        device: ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``.
        dtype: torch dtype.

    Returns:
        SALM model in eval mode on the requested device.
    """
    # Lazy import — surface errors only after subprocess launch.
    from nemo.collections.speechlm2.models import SALM  # type: ignore

    logger.info("Loading Canary-Qwen-2.5B (SALM) from %s", path)

    if os.path.isdir(path):
        # Make sure ``pretrained_llm`` (default: ``Qwen/Qwen3-1.7B``)
        # points at a local checkpoint so that SALM's tokenizer / LLM
        # loader works in offline environments.
        _patch_pretrained_llm_to_local(path)

        # NeMo SALM checkpoints are usually distributed as a single .nemo
        # file plus a HF-style directory.  Prefer a .nemo file if found.
        nemo_files = [f for f in os.listdir(path) if f.endswith(".nemo")]
        if nemo_files:
            nemo_files.sort(
                key=lambda f: os.path.getsize(os.path.join(path, f)),
                reverse=True,
            )
            restore_path = os.path.join(path, nemo_files[0])
            model = SALM.restore_from(
                restore_path=restore_path,
                map_location=torch.device(device),
            )
        else:
            # Pass the directory directly — newer SALM versions accept HF
            # snapshot dirs containing config.yaml + weights.
            model = SALM.from_pretrained(path, map_location=device)
    else:
        # Treat as HuggingFace repo id (e.g. ``nvidia/canary-qwen-2.5b``).
        model = SALM.from_pretrained(path, map_location=device)

    model = model.to(device)
    model.eval()
    if dtype in (torch.float16, torch.bfloat16):
        try:
            model = model.to(dtype)
        except Exception as e:
            logger.warning(
                "Failed to cast SALM model to %s, keeping fp32: %s", dtype, e,
            )

        # NOTE: We intentionally do NOT keep the audio preprocessor in fp32.
        # SALM's conformer encoder weights are now in bf16, so feeding it a
        # fp32 mel-spectrogram produces a dtype mismatch:
        #   "Input type (float) and bias type (c10::BFloat16) should be the
        #    same"
        # at the very first conv2d in subsampling.  Keeping the preprocessor
        # at the same dtype as the rest of the model avoids that.

    return model


def _transcribe_one(model, audio_path, prompt_text, max_new_tokens):
    """Run ASR on a single audio file via SALM-style chat prompting.

    Returns ``(content, raw_text)``.
    """
    audio_locator = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    user_content = f"{prompt_text} {audio_locator}"
    # Use SALM's official high-level API (Example 1 in
    # ``SALM.generate`` docstring): place the audio path inside the
    # prompt turn under the ``audio`` key.  SALM will internally call
    # ``_resolve_audios_in_prompt`` -> Lhotse to load + resample the
    # waveform and produce ``audio_lens``.  Passing ``audios=[path]``
    # directly to ``generate`` would forward a ``str`` list straight
    # into ``perception(input_signal, input_signal_length=None)`` and
    # trigger the "input_signal / processed_signal mutually exclusive"
    # ValueError inside ``AudioPerceptionModule.maybe_preprocess_audio``.
    prompts = [[
        {"role": "user", "content": user_content, "audio": [audio_path]},
    ]]

    with torch.no_grad():
        answer_ids = model.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

    # ``answer_ids`` is List[Tensor] (B=1 here).
    if not len(answer_ids):
        return "", ""
    ids = answer_ids[0]
    if hasattr(ids, "cpu"):
        ids = ids.cpu()
    try:
        raw_text = model.tokenizer.ids_to_text(ids)
    except AttributeError:
        # Some SALM versions expose ``tokenizer.decode``.
        raw_text = model.tokenizer.decode(ids)
    raw_text = raw_text or ""
    return _clean_text(raw_text), raw_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="Local path or HF repo id of the Canary-Qwen-2.5B checkpoint.",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Model dtype",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run the model on (e.g. cuda:0 / cpu)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens generated per request.",
    )
    parser.add_argument(
        "--prompt_text", type=str, default=DEFAULT_ASR_PROMPT,
        help="Default ASR instruction prepended to the audio locator.",
    )
    args = parser.parse_args()

    dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    if args.device == "cpu":
        dtype = torch.float32

    model_path = args.path
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        model_path = os.path.abspath(model_path)

    asr = _load_canary_qwen(model_path, args.device, dtype)
    print(f"Model loaded from checkpoint: {model_path}", flush=True)
    logger.info("Canary-Qwen-2.5B (SALM) loaded successfully")

    while True:
        try:
            prompt = input()
        except EOFError:
            break

        try:
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid request format, must contain '->' but got "
                    f"{prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            payload = json.loads(prompt[anchor + 2:])

            audio_path = payload.get("audio") or payload.get("WavPath")
            if not audio_path:
                print(f"{prefix}Error: 'audio' field is required", flush=True)
                continue
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(audio_path)
            if not os.path.exists(audio_path):
                print(
                    f"{prefix}Error: audio file not found: {audio_path}",
                    flush=True,
                )
                continue

            kwargs = payload.get("kwargs", {}) or {}
            req_prompt = kwargs.pop("prompt_text", None) or args.prompt_text
            req_max_new = int(
                kwargs.pop("max_new_tokens", args.max_new_tokens)
            )

            start_time = time.time()
            text, raw_text = _transcribe_one(
                asr, audio_path, req_prompt, req_max_new,
            )
            elapsed = time.time() - start_time
            logger.info(
                "ASR done in %.2fs, len=%d, audio=%s",
                elapsed, len(text), os.path.basename(audio_path),
            )

            result = json.dumps(
                {
                    "content": text,
                    "raw_text": raw_text,
                },
                ensure_ascii=False,
            )

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 30)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                if retry:
                    logger.debug(
                        "close signal not received within 30s, will emit again"
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    main()
