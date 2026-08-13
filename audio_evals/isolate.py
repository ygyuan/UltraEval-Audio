import atexit
import os
import subprocess
import logging
import time
from functools import wraps

from audio_evals.env_setup import ensure_env

logger = logging.getLogger(__name__)

# Maximum number of times an isolated subprocess may be auto-restarted
# before we give up and propagate the failure.  Without this guard, a
# misconfigured model (e.g. wrong checkpoint path, missing dependency,
# CUDA driver mismatch) would crash on every cold-start and the parent
# process would loop forever trying to restart it -- producing the
# observed 97% fail-rate on omnivoice / minimax_tts_chinese.
# Overridable via env var ``ULTRAEVAL_MAX_RESTART``.
_DEFAULT_MAX_RESTART = int(os.environ.get("ULTRAEVAL_MAX_RESTART", "5"))
# Time (seconds) the subprocess must stay alive after a restart before
# we count it as "successfully restarted".  Anything shorter than this is
# treated as a cold-start crash and counts against ``_DEFAULT_MAX_RESTART``.
_RESTART_GRACE_SEC = float(os.environ.get("ULTRAEVAL_RESTART_GRACE_SEC", "3.0"))


def isolated(
    script_path: str, command_args_attr: str = "command_args", pre_command: str = ""
):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, env_path, requirements_path, *args, gpu_id=None, **kwargs):
            """
            Args:
                env_path: 虚拟环境路径
                requirements_path: 依赖文件路径
                gpu_id: 指定使用的 GPU ID，如 0, 1, 2。
                        如果为 None，则不设置 CUDA_VISIBLE_DEVICES（使用默认行为）
            """
            original_init(self, *args, **kwargs)
            if env_path.endswith("/"):
                env_path = env_path[:-1]

            # 保存 gpu_id 供外部查询
            self._gpu_id = gpu_id

            # 创建虚拟环境并安装依赖（跨线程/跨进程只会真正执行一次）
            ensure_env(env_path, requirements_path, pre_command)

            # 自动检测 Python 版本
            # Use python -c to print only the version string, avoiding
            # bashrc output pollution (e.g. nvm, conda, custom prompts).
            python_version = (
                subprocess.check_output(
                    f"source {env_path}/bin/activate && python -c \"import sys; print('.'.join(map(str, sys.version_info[:2])))\"",
                    shell=True,
                    executable="/bin/bash",
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .splitlines()[-1]
            )
            major_minor = python_version

            # uv-managed python-build-standalone interpreters sometimes keep their
            # shared libpythonX.Y under the underlying install's own lib/ dir,
            # which isn't on the default dynamic linker search path.
            python_base_prefix = subprocess.check_output(
                f"source {env_path}/bin/activate && python -c 'import sys; print(sys.base_prefix)'",
                shell=True,
                executable="/bin/bash",
                text=True,
            ).strip()
            python_lib_dir = f"{python_base_prefix}/lib"

            # 构建 LD_LIBRARY_PATH
            # Include all critical NVIDIA library paths from the venv to ensure
            # pip-installed versions take priority over system-installed ones.
            nvidia_pkg_base = f"{env_path}/lib/python{major_minor}/site-packages/nvidia"
            lib_path = f"{nvidia_pkg_base}/nvjitlink/lib"
            cuda_runtime_lib = f"{nvidia_pkg_base}/cuda_runtime/lib"
            cudnn_lib = f"{nvidia_pkg_base}/cudnn/lib"
            cublas_lib = f"{nvidia_pkg_base}/cublas/lib"
            cufft_lib = f"{nvidia_pkg_base}/cufft/lib"
            # CUDA 13 unified pip layout: nvidia-cu13 ships all CUDA 13 runtime
            # libraries (libcudart.so.13, libcublas.so.13, libnvJitLink.so.13,
            # libcupti.so.13, libnvrtc.so.13, ...) under a single ``cu13/lib``
            # directory. Newer vllm wheels (>=0.22) compiled against CUDA 13
            # depend on these SONAMEs. Adding this directory is harmless for
            # CUDA 12 stacks because the SONAMEs do not collide
            # (e.g. libcudart.so.12 vs libcudart.so.13).
            cu13_lib = f"{nvidia_pkg_base}/cu13/lib"

            cuda_runtime_lib = f"{env_path}/lib/python{major_minor}/site-packages/nvidia/cuda_runtime/lib"

            # 构建命令行参数
            command_args = getattr(self, command_args_attr, {})
            args_str = " ".join(
                [
                    f"--{key} " if value == "" else f"--{key} '{value}'"
                    for key, value in command_args.items()
                ]
            )

            # 构建 CUDA_VISIBLE_DEVICES 设置
            cuda_env = ""
            if gpu_id is not None:
                # gpu_id 由 IsolatedModelPool 注入，仅用于隔离子进程可见的物理
                # GPU。子进程内设备会重新编号（首张卡仍是 cuda:0），因此模型
                # 注册配置不应再指定或根据 gpu_id 改写 device。
                cuda_env = f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for isolated process")

            # 构建完整命令
            command = (
                f"source {env_path}/bin/activate && "
                f"{cuda_env}"
                f"export LD_LIBRARY_PATH={lib_path}:{cuda_runtime_lib}:{cudnn_lib}:{cublas_lib}:{cufft_lib}:{cu13_lib}:{python_lib_dir}:$LD_LIBRARY_PATH && "
                f"export PYTORCH_NVML_BASED_CUDA_CHECK=0 && "

                f"{env_path}/bin/python -u {script_path} {args_str}"
            )
            logger.info(f"Running command: {command}")
            # Save the launch command for potential subprocess restart
            self._launch_command = command
            # Track auto-restarts so that a persistently broken subprocess
            # (e.g. failing model load) eventually surfaces as an error
            # instead of silently crashing on every sample.
            self._restart_count = 0
            self._max_restart = _DEFAULT_MAX_RESTART
            self._last_restart_ts = 0.0

            # Build a clean child env: inherit parent env but force-set the
            # GPU mask so user rc-scripts (e.g. /root/custom.bashrc, nvm,
            # conda) cannot accidentally unset CUDA_VISIBLE_DEVICES before
            # our intra-command `export CUDA_VISIBLE_DEVICES=N` takes
            # effect. This makes the GPU pinning robust even on hosts with
            # aggressive bashrc setups.
            child_env = os.environ.copy()
            if gpu_id is not None:
                child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            child_env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

            self._child_env = child_env
            self.process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash",
                env=child_env,
            )

            # 添加检查进程状态并打印错误信息的方法
            def check_process_status(self_ref):
                """检查进程状态，如果进程已退出则打印所有输出信息"""
                if self_ref.process.poll() is not None:
                    exit_code = self_ref.process.returncode
                    logger.error(f"Process has exited with code: {exit_code}")
                    try:
                        # 读取剩余的输出
                        stdout, stderr = self_ref.process.communicate(timeout=5)
                        if stdout:
                            logger.error(f"Process STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Process STDERR:\n{stderr}")
                    except Exception as e:
                        logger.error(f"Failed to read process output: {e}")
                    return False
                return True

            self.check_process_status = lambda: check_process_status(self)

            def restart_process(self_ref):
                """Restart the subprocess using the saved launch command.

                This is used to recover from subprocess crashes (e.g., GPU OOM,
                segfault, CUDA context corruption, or other unexpected
                terminations).

                Raises:
                    RuntimeError: when the cumulative number of restarts
                        exceeds ``self._max_restart``.  This prevents an
                        infinite restart loop on a persistently broken
                        worker (e.g. wrong model path / missing pkg /
                        driver mismatch).
                """
                # Hard cap: do not keep cold-starting a worker that is
                # clearly broken at the model-load stage.
                if self_ref._restart_count >= self_ref._max_restart:
                    raise RuntimeError(
                        f"Subprocess has already been restarted "
                        f"{self_ref._restart_count} times "
                        f"(>= max {self_ref._max_restart}); refusing to "
                        f"restart again. Check the subprocess STDERR "
                        f"above for the real root cause."
                    )

                self_ref._restart_count += 1
                logger.warning(
                    "Restarting subprocess (attempt %d/%d)...",
                    self_ref._restart_count,
                    self_ref._max_restart,
                )
                # Terminate old process if still running
                if self_ref.process.poll() is None:
                    self_ref.process.terminate()
                    try:
                        self_ref.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self_ref.process.kill()
                        self_ref.process.wait(timeout=5)

                # Exponential-ish back-off so we do not hammer the GPU /
                # filesystem when the previous restart died immediately.
                # 0.5s, 1s, 2s, 4s, 8s, ... capped at 30s.
                backoff = min(30.0, 0.5 * (2 ** (self_ref._restart_count - 1)))
                if backoff > 0:
                    time.sleep(backoff)

                # Start a new subprocess with the same command
                self_ref.process = subprocess.Popen(
                    self_ref._launch_command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    executable="/bin/bash",
                    env=self_ref._child_env,
                )
                self_ref._last_restart_ts = time.time()
                logger.info(
                    "Subprocess restarted successfully (pid=%d, attempt %d/%d)",
                    self_ref.process.pid,
                    self_ref._restart_count,
                    self_ref._max_restart,
                )

                # Quick sanity check: if the freshly-spawned process dies
                # within ``_RESTART_GRACE_SEC``, treat the restart itself
                # as a cold-start failure (the next call will count
                # against ``_max_restart``) but do NOT block forever
                # waiting for model load -- we only want to catch the
                # *immediate* import-time / arg-parse crashes.
                early_exit_check = min(0.5, _RESTART_GRACE_SEC)
                time.sleep(early_exit_check)
                if self_ref.process.poll() is not None:
                    exit_code = self_ref.process.returncode
                    try:
                        _, stderr_tail = self_ref.process.communicate(
                            timeout=5
                        )
                        if stderr_tail:
                            logger.error(
                                "Subprocess died %.1fs after restart "
                                "(exit code: %d). STDERR tail:\n%s",
                                early_exit_check,
                                exit_code,
                                stderr_tail[-4000:],
                            )
                    except Exception:
                        logger.error(
                            "Subprocess died %.1fs after restart "
                            "(exit code: %d).",
                            early_exit_check,
                            exit_code,
                        )

            self.restart_process = lambda: restart_process(self)

            def ensure_process_alive(self_ref):
                """Check if subprocess is alive; restart it if it has terminated.

                Returns:
                    True if the process was already alive, False if it was
                    restarted (caller may want to log this).

                Raises:
                    RuntimeError: If the subprocess cannot be restarted, OR
                        if the subprocess was killed by the user (SIGINT /
                        SIGTERM) — in that case we propagate KeyboardInterrupt
                        instead of silently restarting a "zombie" worker.
                """
                if self_ref.process.poll() is None:
                    return True  # Process is alive

                exit_code = self_ref.process.returncode

                # Negative exit codes correspond to ``-signal`` on POSIX.
                # If the user hit Ctrl+C (SIGINT, exit_code == -2) or the
                # parent sent SIGTERM (exit_code == -15) we should stop the
                # whole evaluation, NOT auto-restart a fresh subprocess
                # behind the user's back (which previously produced "zombie"
                # workers consuming GPU memory after Ctrl+C).
                if exit_code in (-2, -15):
                    logger.warning(
                        "Subprocess was terminated by signal "
                        "(exit code: %d, SIGINT/SIGTERM). Treating this as a "
                        "user-initiated shutdown — NOT auto-restarting.",
                        exit_code,
                    )
                    raise KeyboardInterrupt(
                        f"Subprocess terminated by signal "
                        f"(exit code: {exit_code})"
                    )

                logger.error(
                    f"Subprocess has terminated unexpectedly (exit code: {exit_code}). "
                    f"Attempting automatic restart..."
                )
                # Drain remaining output for diagnostics
                try:
                    stdout, stderr = self_ref.process.communicate(timeout=5)
                    if stdout:
                        logger.error(f"Dead process STDOUT:\n{stdout[-2000:]}")
                    if stderr:
                        logger.error(f"Dead process STDERR:\n{stderr[-2000:]}")
                except Exception as e:
                    logger.warning(f"Could not read dead process output: {e}")

                try:
                    self_ref.restart_process()
                    return False  # Process was dead but has been restarted
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restart subprocess after crash (exit code: {exit_code}): {e}"
                    )

            self.ensure_process_alive = lambda: ensure_process_alive(self)

            # 注册清理函数
            def cleanup():
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                else:
                    # 进程已退出，打印输出信息
                    exit_code = self.process.returncode
                    logger.info(
                        f"Process already exited with code: {exit_code} "
                        f"(gpu_id={self._gpu_id})"
                    )
                    try:
                        stdout, stderr = self.process.communicate(timeout=5)
                        if stdout:
                            logger.info(f"Final STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Final STDERR:\n{stderr}")
                    except Exception as e:
                        logger.warning(f"Could not read final output: {e}")

            atexit.register(cleanup)

        cls.__init__ = new_init
        return cls

    return decorator
