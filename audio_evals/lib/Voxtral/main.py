# pyright: reportMissingImports=false
"""
Voxtral 4B TTS subprocess launcher.

Boots a `vllm serve <model> --omni` HTTP server inside an isolated venv and
reports the base URL on stdout (`PORT:<base_url>`) so the parent
``audio_evals.models.TTS.voxtral_tts.VoxtralTTS`` wrapper can issue
`/v1/audio/speech` requests against it.

Reference:
    https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
"""
import argparse
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_server_process = None
LOCAL_HOST = "127.0.0.1"


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCAL_HOST, 0))
        return sock.getsockname()[1]


def cleanup_server():
    global _server_process
    if _server_process is None or _server_process.poll() is not None:
        return

    logger.info("Cleaning up Voxtral vLLM server process group")
    try:
        pgid = os.getpgid(_server_process.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            _server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _server_process.wait(timeout=5)
    except Exception as exc:
        logger.warning("Failed to cleanup Voxtral server gracefully: %s", exc)
        try:
            _server_process.kill()
        except Exception:
            pass


def signal_handler(signum, frame):
    logger.info("Received signal %s, shutting down...", signum)
    cleanup_server()
    sys.exit(0)


def wait_for_server(base_url: str, timeout: int = 1800) -> None:
    health_urls = [
        f"{base_url}/health",
        f"{base_url}/v1/models",
    ]
    start = time.time()

    while time.time() - start < timeout:
        if _server_process is not None and _server_process.poll() is not None:
            raise RuntimeError(
                f"Voxtral vLLM server exited early with code "
                f"{_server_process.returncode}"
            )
        for url in health_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("Voxtral vLLM server is ready via %s", url)
                    return
            except requests.RequestException:
                pass
        time.sleep(3)

    raise TimeoutError(
        f"Voxtral vLLM server not ready within {timeout} seconds"
    )


def start_server(args) -> str:
    global _server_process

    port = find_available_port()
    base_url = f"http://{LOCAL_HOST}:{port}"
    # vllm-omni provides the `--omni` flag to `vllm serve`. Prefer the
    # `vllm-omni` console script because vllm-omni 0.18.0 ships its own
    # CLI (`vllm_omni.entrypoints.cli.main`) which is the only entry point
    # that registers `--omni`. The plain `vllm` script may have been
    # installed by vllm itself and does not know about `--omni`.
    bin_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(bin_dir, "vllm-omni")
    if os.path.exists(candidate):
        vllm_bin = candidate
    else:
        vllm_bin = os.path.join(bin_dir, "vllm")
        if not os.path.exists(vllm_bin):
            vllm_bin = "vllm"
    cmd = [
        vllm_bin,
        "serve",
        args.path,
        "--omni",
        "--host",
        LOCAL_HOST,
        "--port",
        str(port),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.tensor_parallel_size and args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization:
        cmd.extend([
            "--gpu-memory-utilization", str(args.gpu_memory_utilization)
        ])
    if args.extra_args:
        cmd.extend([t for t in args.extra_args.split() if t])

    logger.info("Starting Voxtral vLLM server: %s", " ".join(cmd))
    # Forward both stdout and stderr from vllm to *our* stderr (fd 2). The
    # parent isolate process pipes our stderr separately and prints it on
    # cleanup, so vllm errors (e.g. version-mismatch / OOM) become visible
    # in the eval log instead of being silently swallowed by DEVNULL.
    # We deliberately do NOT pipe to our stdout, because stdout is reserved
    # for the `PORT:<base_url>` protocol line consumed by the parent.
    _server_process = subprocess.Popen(
        cmd,
        stdout=sys.stderr,
        stderr=sys.stderr,
        preexec_fn=os.setsid,
    )

    wait_for_server(base_url, timeout=args.startup_timeout)
    return base_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path or HF id of the Voxtral TTS model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype passed through to vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.0)
    parser.add_argument("--extra_args", type=str, default="")
    parser.add_argument("--startup_timeout", type=int, default=1800)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_server)

    try:
        base_url = start_server(args)
    except Exception as exc:
        print(
            f"Failed to initialize Voxtral vLLM server: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

    print(f"PORT:{base_url}", flush=True)
    logger.info("Voxtral main.py launched service at %s", base_url)

    try:
        _server_process.wait()
    finally:
        cleanup_server()
