import atexit
import os
import subprocess
import logging
from functools import wraps

logger = logging.getLogger(__name__)


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

            # 创建虚拟环境
            if not os.path.exists(env_path):
                # Try uv first, fall back to standard venv
                uv_available = subprocess.run(
                    ["which", "uv"], capture_output=True
                ).returncode == 0
                if uv_available:
                    res = subprocess.run(["uv", "venv", env_path, "--python", "3.10", "--allow-existing"])
                else:
                    import sys
                    res = subprocess.run([sys.executable, "-m", "venv", env_path])
                if res.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create virtual environment: {res.stderr}"
                    )

            # 安装依赖
            uv_available = subprocess.run(
                ["which", "uv"], capture_output=True
            ).returncode == 0
            # Create a constraint file to limit setuptools version in pip's build isolation
            constraint_file = os.path.join(env_path, "build_constraints.txt")
            with open(constraint_file, "w") as f:
                f.write("setuptools<81\n")

            pip_find_links = os.environ.get("ULTRAEVAL_PIP_FIND_LINKS", "").strip()
            pip_no_index = os.environ.get("ULTRAEVAL_PIP_NO_INDEX", "").strip() == "1"
            pip_index_url = os.environ.get("ULTRAEVAL_PIP_INDEX_URL", "").strip()
            pip_extra_index_url = os.environ.get(
                "ULTRAEVAL_PIP_EXTRA_INDEX_URL", ""
            ).strip()

            pip_option_parts = []
            if pip_no_index:
                pip_option_parts.append("--no-index")
            if pip_find_links:
                pip_option_parts.append(f"--find-links '{pip_find_links}'")
            if pip_index_url:
                pip_option_parts.append(f"--index-url '{pip_index_url}'")
            if pip_extra_index_url:
                pip_option_parts.append(
                    f"--extra-index-url '{pip_extra_index_url}'"
                )
            pip_options = " ".join(pip_option_parts)

            if uv_available:
                install_cmd = (
                    f"source {env_path}/bin/activate &&"
                    f"{pre_command + '&& ' if pre_command else ''}"
                    f" uv pip install {pip_options} setuptools\\<81 &&"
                    f" uv pip install {pip_options} -r {requirements_path}"
                )
            else:
                # Replace "uv pip" with "pip" in pre_command if uv is not available
                actual_pre_command = pre_command.replace("uv pip", "pip") if pre_command else ""
                install_cmd = (
                    f"source {env_path}/bin/activate &&"
                    f"{actual_pre_command + '&& ' if actual_pre_command else ''}"
                    f" pip install {pip_options} --upgrade 'setuptools<81' &&"
                    f" PIP_CONSTRAINT={constraint_file} pip install {pip_options} -r {requirements_path}"
                )
            result = subprocess.run(
                install_cmd,
                shell=True,
                check=True,
                executable="/bin/bash",
            )
            if result.returncode != 0:
                raise RuntimeError(f"Dependency installation failed: {result.stderr}")

            # 自动检测 Python 版本
            python_version = (
                subprocess.check_output(
                    f"source {env_path}/bin/activate && python --version",
                    shell=True,
                    executable="/bin/bash",
                    text=True,
                )
                .strip()
                .split()[1]
            )
            major_minor = ".".join(python_version.split(".")[:2])

            # 构建 LD_LIBRARY_PATH
            lib_path = (
                f"{env_path}/lib/python{major_minor}/site-packages/nvidia/nvjitlink/lib"
            )

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
                cuda_env = f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for isolated process")

            # 构建完整命令
            command = (
                f"source {env_path}/bin/activate && "
                f"{cuda_env}"
                f"export LD_LIBRARY_PATH={lib_path} && "
                f"export PYTORCH_NVML_BASED_CUDA_CHECK=0 && "
                f"{env_path}/bin/python -u {script_path} {args_str}"
            )
            logger.info(f"Running command: {command}")
            # Save the launch command for potential subprocess restart
            self._launch_command = command
            self.process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash",
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
                segfault, or other unexpected terminations).
                """
                logger.warning("Restarting subprocess...")
                # Terminate old process if still running
                if self_ref.process.poll() is None:
                    self_ref.process.terminate()
                    try:
                        self_ref.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self_ref.process.kill()
                        self_ref.process.wait(timeout=5)

                # Start a new subprocess with the same command
                self_ref.process = subprocess.Popen(
                    self_ref._launch_command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    executable="/bin/bash",
                )
                logger.info("Subprocess restarted successfully (pid=%d)", self_ref.process.pid)

            self.restart_process = lambda: restart_process(self)

            def ensure_process_alive(self_ref):
                """Check if subprocess is alive; restart it if it has terminated.

                Returns:
                    True if the process was already alive, False if it was
                    restarted (caller may want to log this).

                Raises:
                    RuntimeError: If the subprocess cannot be restarted.
                """
                if self_ref.process.poll() is None:
                    return True  # Process is alive

                exit_code = self_ref.process.returncode
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
                        self.process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                else:
                    # 进程已退出，打印输出信息
                    exit_code = self.process.returncode
                    logger.info(f"Process already exited with code: {exit_code}")
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
