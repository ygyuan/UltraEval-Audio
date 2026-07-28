import argparse
import gc
import json
import select
import sys
import tempfile

import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


device = "cuda"


# Names of per-request intermediate objects we want to drop from the loop's
# local scope after each iteration. Anything holding a CUDA tensor should be
# listed here; missing names are silently ignored.
_PER_REQUEST_LOCALS = (
    "inputs",
    "audios",
    "images",
    "videos",
    "output_ids",
    "generated",
    "text_ids",
    "audio",
    "raw",
)


def _release_cuda_memory(local_scope):
    """Best-effort release of per-request tensors and CUDA cache.

    We do this after every request (success or failure) so that long-running
    evaluations do not accumulate fragmentation / stale allocations in the
    caching allocator, which was the root cause of the 66 GiB creeping usage
    and eventual OOM observed on 30k+ samples.
    """
    for name in _PER_REQUEST_LOCALS:
        if name in local_scope:
            try:
                del local_scope[name]
            except Exception:
                pass
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_model(path, **kwargs):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        **kwargs,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(path)
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        help="Speaker name for speech generation",
    )
    parser.add_argument(
        "--thinker_max_new_tokens",
        type=int,
        default=1024,
        help=(
            "Max new tokens for the thinker stage. The official default is "
            "very small (256), which is easily exhausted by Qwen3-Omni's "
            "internal <think>...</think> stream and yields an empty final "
            "answer; bump it to 1024 by default."
        ),
    )
    parser.add_argument(
        "--talker_max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for the talker (audio) stage.",
    )
    config = parser.parse_args()
    model, processor = load_model(config.path)
    print("Model loaded from checkpoint: {}".format(config.path), flush=True)

    while True:
        # Prefix used by the parent to correlate request/response; declared
        # outside the try so we can still emit a well-formed error line if
        # anything goes wrong after we managed to parse it.
        prefix = None
        try:
            prompt = input()

            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            conversation = json.loads(prompt[anchor + 2 :])

            # Set whether to use audio in video
            USE_AUDIO_IN_VIDEO = True

            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False,
                enable_thinking=True,
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            if config.speech:
                with torch.inference_mode():
                    text_ids, audio = model.generate(
                        **inputs,
                        speaker=config.speaker,
                        use_audio_in_video=USE_AUDIO_IN_VIDEO,
                        thinker_max_new_tokens=config.thinker_max_new_tokens,
                        talker_max_new_tokens=config.talker_max_new_tokens,
                    )
                text = processor.batch_decode(
                    text_ids[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if audio is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(
                            f.name,
                            audio.reshape(-1).detach().cpu().numpy(),
                            samplerate=24000,
                        )
                        retry = 3
                        while retry:
                            retry -= 1
                            print(
                                prefix + json.dumps({"text": text[0], "audio": f.name}),
                                flush=True,
                            )
                            rlist, _, _ = select.select([sys.stdin], [], [], 1)
                            if rlist:
                                finish = sys.stdin.readline().strip()
                                if finish == "{}close".format(prefix):
                                    break
                            print("not found close signal, will emit again", flush=True)
                else:
                    retry = 3
                    while retry:
                        retry -= 1
                        print(prefix + json.dumps({"text": text[0]}), flush=True)
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)
            else:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        use_audio_in_video=USE_AUDIO_IN_VIDEO,
                        return_audio=False,
                        thinker_max_new_tokens=config.thinker_max_new_tokens,
                        talker_max_new_tokens=config.talker_max_new_tokens,
                    )
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not text or not text[0].strip():
                    # Empty decode usually means the thinker budget was
                    # exhausted before any visible answer was produced.
                    raw = processor.batch_decode(
                        generated,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    print(
                        "Warning: empty decoded text. generated_len={}, "
                        "thinker_max_new_tokens={}, raw_with_special={!r}".format(
                            generated.shape[-1],
                            config.thinker_max_new_tokens,
                            (raw[0] if raw else "")[:500],
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                retry = 3
                while retry:
                    retry -= 1
                    print(prefix + json.dumps({"text": text[0]}), flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == "{}close".format(prefix):
                            break
                    print("not found close signal, will emit again", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            # Dedicated OOM branch: do NOT kill the subprocess. Report the
            # failure back to the parent for this single sample and let the
            # loop continue with a freshly-cleaned allocator so the next
            # request has a chance to succeed.
            import traceback

            traceback.print_exc()
            err_msg = "CUDA OOM: {}".format(str(e).splitlines()[0] if str(e) else "unknown")
            if prefix is not None:
                # Reply on the same prefix so the parent unblocks its read
                # loop; we send an Error: line which the parent recognises
                # as a per-sample failure.
                print("Error:" + err_msg, flush=True)
            else:
                print("Error:" + err_msg, flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            msg = str(e)
            # Some CUDA OOMs surface as plain RuntimeError instead of the
            # dedicated OutOfMemoryError type; treat them the same way so we
            # do not tear down the subprocess.
            lowered = msg.lower()
            if "out of memory" in lowered or "cuda oom" in lowered:
                print("Error: CUDA OOM: {}".format(msg.splitlines()[0] if msg else "unknown"), flush=True)
            else:
                print("Error:" + msg, flush=True)
        finally:
            # Always release per-request tensors and empty the CUDA cache.
            # Doing this on every iteration (success or failure) is what
            # keeps the process's GPU memory footprint bounded over tens of
            # thousands of samples.
            _release_cuda_memory(locals())
