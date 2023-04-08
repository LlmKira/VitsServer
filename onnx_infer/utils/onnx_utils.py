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


class RunONNX(object):
    def __init__(self, model=None,
                 providers=None):
        self.ort_session = None
        if model:
            self.load(model, providers=providers)

    def load(self,
             model,
             providers=None
             ):
        # 如果是 ByteIO 类，则转换为 bytes
        if providers is None:
            providers = ['CPUExecutionProvider']
        if hasattr(model, "getvalue"):
            model = model.getvalue()
        # 创造运行时
        self.ort_session = ort.InferenceSession(model, providers=providers)

    def run(self, model_input):
        outputs = self.ort_session.run(
            None,
            input_feed=model_input
        )
        return outputs
