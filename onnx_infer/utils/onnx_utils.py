import random

import numpy as np
import onnxruntime as ort


# import torch.backends.cuda


def set_random_seed(seed=0):
    import torch.backends.cudnn
    ort.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


def runonnx(model, **kwargs):
    # 如果是 ByteIO 类，则转换为 bytes
    if hasattr(model, "getvalue"):
        model = model.getvalue()
    # 创造运行时
    ort_session = ort.InferenceSession(model)
    outputs = ort_session.run(
        None,
        kwargs
    )
    return outputs
