import base64
import functools
import glob
import importlib
import json
import logging
import os
import re
import time
import typing

import pandas as pd

from audio_evals.base import EarlyStop

logger = logging.getLogger(__name__)


def retry(max_retries=3, default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except EarlyStop as e:
                    if default is not None:
                        return default
                    raise e
                except Exception as e:
                    last_exception = e
                    logger.error(f"retry after: {e}")
                    time.sleep(1)  # 可选：添加延迟
            if default is not None:
                return default
            raise last_exception  # 抛出最后一次捕获的异常

        return wrapper

    return decorator


def make_object(class_name: str, **kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
    module_name, qualname = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, qualname)
    return cls(**kwargs)


MIME_TYPE_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    # 可以根据需要添加更多文件格式的支持
}


def get_base64_from_file(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded = base64.b64encode(file_content).decode("utf-8")
        return base64_encoded


def convbase64(file_path):
    """
    将语音文件转换为包含MIME类型的Base64编码数据URI。

    :param file_path: 语音文件的路径
    :return: 包含MIME类型的Base64编码数据URI
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type = MIME_TYPE_MAP.get(file_extension)

        if not mime_type:
            raise ValueError(f"Unsupported file format: {file_extension}")
        base64_encoded = get_base64_from_file(file_path)
        data_uri = f"data:{mime_type};base64,{base64_encoded}"
        return data_uri
    except Exception as e:
        print(f"Error converting file to Base64: {e}")
        return None


def decode_base64_to_file(data_uri, output_path):
    """
    将包含MIME类型的Base64编码数据URI转化为文件并保存。

    :param data_uri: 包含MIME类型的Base64编码数据URI
    :param output_path: 输出文件的路径
    """
    try:
        # 分离MIME类型和Base64编码部分
        if not data_uri.startswith("data:"):
            raise ValueError("Invalid data URI format")

        header, base64_data = data_uri.split(",", 1)

        # 从头部解析MIME类型
        mime_type = header.split(";")[0][5:]
        file_extension = None

        # 反向查找 MIME 类型映射的文件扩展名
        for ext, mtype in MIME_TYPE_MAP.items():
            if mtype == mime_type:
                file_extension = ext
                break

        if not file_extension:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        # 如果输出路径没有扩展名，添加一个默认扩展名
        if not os.path.splitext(output_path)[1]:
            output_path += file_extension

        # 解码 Base64 数据
        file_data = base64.b64decode(base64_data)

        # 写入文件
        with open(output_path, "wb") as file:
            file.write(file_data)

        print(f"File successfully saved to {output_path}")
    except Exception as e:
        print(f"Error decoding Base64 to file: {e}")


def clean_illegal_chars(text):
    if isinstance(text, str):
        # 移除控制字符（ASCII码小于32的字符）
        return re.sub(r"[\x00-\x1F\x7F]", "", text)
    return text


def _read_jsonl_robust(path: str) -> pd.DataFrame:
    """Read a ``.jsonl`` file into a DataFrame without triggering ujson's
    ``Value is too big!`` overflow on integers larger than int64.

    ``pd.read_json(..., lines=True)`` goes through the ujson parser, which
    rejects integers outside ``[-2**63, 2**63)``. Some datasets embed very
    large numeric IDs (or raw 19+ digit strings that look numeric) inside
    the recorder log; we therefore fall back to the Python ``json`` module
    which handles arbitrary-precision ints natively.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skip malformed jsonl line %d in %s: %s", lineno, path, e
                )
    return pd.DataFrame(records)


def merge_data4view(
    quiz: typing.List[dict], eval_df: typing.Union[pd.DataFrame, str], save_name
):
    quiz = pd.DataFrame(quiz)
    quiz["id"] = range(len(quiz))
    if isinstance(eval_df, str):
        try:
            eval_df = pd.read_json(eval_df, lines=True)
        except (ValueError, OverflowError) as e:
            # ujson rejects ints outside int64 range with "Value is too big!".
            # Fall back to the stdlib json parser, which handles big ints.
            logger.warning(
                "pd.read_json failed (%s); falling back to stdlib json for %s",
                e,
                eval_df,
            )
            eval_df = _read_jsonl_robust(eval_df)

    def concat_gdf(gdf):
        r = {}
        for i in gdf.index:
            if "content" in gdf.loc[i, "data"]:
                r[gdf.loc[i, "type"]] = gdf.loc[i, "data"]["content"]
            else:
                r.update(gdf.loc[i, "data"])
        return r

    real_eval = eval_df.groupby("id").apply(concat_gdf).apply(pd.Series)
    real_eval = real_eval.reset_index()
    df = pd.merge(quiz, real_eval, on="id", how="left")
    df = df.map(clean_illegal_chars)
    try:
        df.to_excel(save_name, index=False)
    except Exception as e:
        # Don't let a visualisation-only failure (e.g. xlsx cell-size limit,
        # illegal chars) wipe out the whole evaluation run.
        logger.error("Failed to write xlsx view %s: %s", save_name, e)
        fallback = os.path.splitext(save_name)[0] + ".csv"
        try:
            df.to_csv(fallback, index=False)
            logger.warning("Wrote CSV fallback view to %s", fallback)
        except Exception as e2:
            logger.error("Also failed to write CSV fallback %s: %s", fallback, e2)


def find_latest_jsonl(directory):
    """Find the most recently modified .jsonl file in directory"""
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        return None
    # Get file with latest modification time
    latest_file = max(jsonl_files, key=os.path.getmtime)
    return latest_file
