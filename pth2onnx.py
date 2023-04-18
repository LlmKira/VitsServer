import io
import os
from typing import Union

import librosa
import torch

import utils
from onnx_infer import onnx_infer
from onnx_infer.infer import commons
from onnx_infer.utils.onnx_utils import RunONNX
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.detach().numpy()


class VitsExtractor(object):
    @staticmethod
    def write_out(model_path, obj):
        """
        创建一个和模型同名的 onnx 文件。写入 ByteIO
        """
        import pathlib
        model_path = pathlib.Path(model_path)
        onnx_path = model_path.parent / f'{model_path.stem}.onnx'
        with open(onnx_path, 'wb') as f:
            f.write(obj.getvalue())

    def convert_model(self, json_path: str,
                      model_path: str,
                      write_down: Union[bool, str] = None,
                      providers=None,
                      ) -> io.BytesIO:
        # Load pa from JSON file
        if providers is None:
            # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if utils.get_device() == "cpu":
                providers = ['CPUExecutionProvider']
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        hps = utils.get_hparams_from_file(json_path)

        # Get symbols and initialize synthesizer model
        symbols = hps.symbols if "symbols" in hps else []
        net_g = onnx_infer.SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)

        # Load model checkpoint and set to evaluation mode
        _ = utils.load_checkpoint(model_path, net_g, None)
        net_g.forward = net_g.export_forward
        _ = net_g.eval()
        scales = torch.FloatTensor([0.667, 1.0, 0.8])
        # make triton dynamic shape happy
        scales = scales.unsqueeze(0)

        onnx_model = io.BytesIO()
        if symbols:
            seq = torch.randint(low=0, high=len(symbols), size=(1, 10), dtype=torch.long)
            seq_len = torch.IntTensor([seq.size(1)]).long()
            sid = torch.IntTensor([0]).long()
        else:
            hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
            audio16000, sampling_rate = librosa.load("sample.wav", sr=16000, mono=True)
            seq = hubert.units(torch.FloatTensor(audio16000).unsqueeze(0).unsqueeze(0).to("cpu"))
            # seq = torch.randint(low=0, high=1, size=(1, 10), dtype=torch.long)
            seq_len = torch.IntTensor([seq.size(1)]).long()
            sid = torch.IntTensor([0]).long()
            # seq = hubert.units(torch.FloatTensor("").unsqueeze(0).unsqueeze(0))
            # seq_len = torch.IntTensor([seq.size(1)]).long()
        dummy_input = (seq, seq_len, scales, sid)
        torch.onnx.export(model=net_g,
                          args=dummy_input,
                          f=onnx_model,
                          input_names=['input', 'input_lengths', 'scales', 'sid'],
                          output_names=['output'],
                          dynamic_axes={
                              'input': {
                                  0: 'batch',
                                  1: 'phonemes'
                              },
                              'input_lengths': {
                                  0: 'batch'
                              },
                              'scales': {
                                  0: 'batch'
                              },
                              'sid': {
                                  0: 'batch'
                              },
                              'output': {
                                  0: 'batch',
                                  1: 'audio',
                                  2: 'audio_length'
                              }
                          },
                          opset_version=13,
                          verbose=False)

        # Verify onnx precision
        torch_output = net_g(seq, seq_len, scales, sid)
        ort_inputs = {
            'input': to_numpy(seq),
            'input_lengths': to_numpy(seq_len),
            'scales': to_numpy(scales),
            'sid': to_numpy(sid),
        }
        if not symbols:
            # TODO 检查模型结构，似乎无法正常导出 Hubert 模型
            ort_inputs.pop("sid")
        onnx_output = RunONNX(model=onnx_model, providers=providers).run(model_input=ort_inputs)
        # Convert PyTorch model to ONNX format
        if write_down:
            if type(write_down) == str:
                with open(write_down, 'wb') as f:
                    f.write(onnx_model.getvalue())
            else:
                self.write_out(model_path, onnx_model)
        # Release memory by deleting PyTorch model
        del net_g
        return onnx_model

    def warp_pth(self, model_config_path: str, model_path: str = None, return_bytes: bool = False) -> Union[bytes, str]:
        import pathlib
        model_config_path = pathlib.Path(model_config_path)
        if model_config_path.suffix != ".json":
            raise ValueError("The model config path must end with .json")
        if not model_config_path.exists():
            raise ValueError("The model config path does not exist")
        # ONNX
        onnx_model_path = model_config_path.parent / f'{model_config_path.stem}.onnx'
        if model_path:
            model_path = pathlib.Path(model_path)
            if model_path.suffix == ".onnx" and model_path.exists():
                # 如果是 .onnx 则直接返回
                onnx_model_path = model_path

        # PTH
        pth_model_path = model_config_path.parent / f'{model_config_path.stem}.pth'
        # 去掉 .json
        if pathlib.Path(onnx_model_path).exists():
            if return_bytes:
                with open(onnx_model_path, 'rb') as f:
                    return f.read()
            return str(onnx_model_path)
        if pathlib.Path(pth_model_path).exists():
            onnx_model_byte = self.convert_model(json_path=str(model_config_path), model_path=str(pth_model_path),
                                                 write_down=True)
            if return_bytes:
                return onnx_model_byte.getvalue()
            return str(onnx_model_path)
        if True:
            raise ValueError("The model files do not exist")


if __name__ == "__main__":
    model = VitsExtractor().warp_pth(
        model_config_path="model/1374_epochs.json",
        model_path="model/1374_epochs.pth",
        return_bytes=True
    )
    # 测试类型
    print(type(model))
    # 导入 onnxruntime 库测试是否可以初始化运行时
    import onnxruntime as ort

    print("onnxruntime version:", ort.__version__)
    _model = ort.InferenceSession(model)
    _model.get_outputs()
    del _model
