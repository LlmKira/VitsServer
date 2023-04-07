# -*- coding: utf-8 -*-
import logging
import os
from typing import Dict, Union

import fasttext
import requests

logger = logging.getLogger(__name__)
models = {"low_mem": None, "high_mem": None}
FTLANG_CACHE = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")

try:
    # silences warnings as the package does not properly use the python 'warnings' package
    # see https://github.com/facebookresearch/fastText/issues/1056
    fasttext.FastText.eprint = lambda *args, **kwargs: None
except Exception as e:
    pass


def check_model(name: str) -> bool:
    # 查看模型是否为 0 字节
    target_path = os.path.join(FTLANG_CACHE, name)
    if os.path.exists(target_path):
        if os.path.getsize(target_path) > 0:
            return True
        else:
            # 0 字节，删除
            os.remove(target_path)
    return False


def download_model(name: str) -> str:
    target_path = os.path.join(FTLANG_CACHE, name)
    if not os.path.exists(target_path):
        logger.info(f"Downloading {name} model ...")
        url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{name}"  # noqa
        os.makedirs(FTLANG_CACHE, exist_ok=True)
        with open(target_path, "wb") as fp:
            response = requests.get(url)
            fp.write(response.content)
        logger.info(f"Downloaded.")
    return target_path


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            check_model("lid.176.ftz")
            model_path = download_model("lid.176.ftz")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            check_model("lid.176.ftz")
            model_path = download_model("lid.176.bin")
            model = fasttext.load_model(model_path)
            models["high_mem"] = model
        return model


def detect(text: str, low_memory=False) -> Dict[str, Union[str, float]]:
    model = get_or_load_model(low_memory)
    labels, scores = model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
