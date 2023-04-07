# -*- coding: utf-8 -*-
# @Time    : 2023/4/8 上午1:10
# @Author  : sudoskys
# @File    : onnx_types.py
# @Software: PyCharm
from typing import Any

from pydantic import BaseModel


class VitsBase(BaseModel):
    dp: Any
    flow: Any
    dec: Any
    enc_p: Any

    class Config:
        arbitrary_types_allowed = True
