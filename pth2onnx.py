import io
from typing import Tuple

import torch

import utils
from onnx_infer import onnx_models
from onnx_infer.infer import commons
from onnx_infer.onnx_types import VitsBase
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class VitsExtractor(object):
    @staticmethod
    def write_out(model_path, obj, model_name):
        """
        创建一个和模型同名同目录的文件夹，然后将 ByteIO 对象写入文件夹内的文件，使用 pathlib
        """
        import pathlib

        # Create a folder with the same name as the model
        folder = pathlib.Path(model_path).parent / pathlib.Path(model_path).stem
        folder.mkdir(parents=True, exist_ok=True)

        # Write the ByteIO object to the folder
        with open(folder / model_name, "wb") as f:
            f.write(obj.getvalue())

    def convert_model(self, json_path: str,
                      model_path: str,
                      write_to_folder: bool = None) -> Tuple[io.BytesIO, io.BytesIO, io.BytesIO, io.BytesIO]:
        # Load pa from JSON file
        hps = utils.get_hparams_from_file(json_path)

        # Get symbols and initialize synthesizer model
        symbols = hps.symbols
        net_g = onnx_models.SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)

        # Load model checkpoint and set to evaluation mode
        _ = net_g.eval()
        _ = utils.load_checkpoint(model_path, net_g)

        # Convert text to tensor and synthesize speech
        text1 = get_text("ありがとうございます", hps)
        stn_tst = text1
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            sid = torch.tensor([0])
            enc_p_io, dp_io, flow_io, dec_io = net_g(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1
            )

        # Convert PyTorch model to ONNX format
        if write_to_folder:
            self.write_out(model_path, enc_p_io, "enc_p.onnx")
            self.write_out(model_path, dp_io, "dp.onnx")
            self.write_out(model_path, flow_io, "flow.onnx")
            self.write_out(model_path, dec_io, "dec.onnx")
        # Release memory by deleting PyTorch model
        del net_g

        return enc_p_io, dp_io, flow_io, dec_io

    def warp_pth(self, model_config_path: str, model_path: str = None) -> VitsBase:
        # 输入 some/config.pth.json 检查是否存在 some/config.pth 和 some/config
        import pathlib
        if pathlib.Path(model_config_path).suffix != ".json":
            raise ValueError("The model config path must end with .json")
        # 去掉 .json
        if not model_path:
            model_path = pathlib.Path(model_config_path).with_suffix("")
        folder = pathlib.Path(model_path).parent / pathlib.Path(model_path).stem
        if not folder.exists() and not model_path.exists():
            raise ValueError("The model folder does not exist")
        # 检查是否存在 enc_p.onnx, dp.onnx, flow.onnx, dec.onnx
        if folder.exists():
            enc_p_path = folder / "enc_p.onnx"
            dp_path = folder / "dp.onnx"
            flow_path = folder / "flow.onnx"
            dec_path = folder / "dec.onnx"
            if not enc_p_path.exists() or not dp_path.exists() or not flow_path.exists() or not dec_path.exists():
                raise ValueError("The model files do not exist")
            return VitsBase(
                enc_p=str(enc_p_path.absolute()),
                dp=str(dp_path.absolute()),
                flow=str(flow_path.absolute()),
                dec=str(dec_path.absolute())
            )
        if model_path.exists():
            enc_p_io, dp_io, flow_io, dec_io = self.convert_model(
                json_path="model/1374_epochs.pth.json",
                model_path="model/1374_epochs.pth",
                write_to_folder=True
            )
            return VitsBase(enc_p=enc_p_io, dp=dp_io, flow=flow_io, dec=dec_io)
        if True:
            raise ValueError("The model files do not exist")


if __name__ == "__main__":
    enc_p_io, dp_io, flow_io, dec_io = VitsExtractor().convert_model(
        json_path="model/1374_epochs.pth.json",
        model_path="model/1374_epochs.pth",
        write_to_folder=True
    )
    # 测试类型
    print(type(enc_p_io))
    # 导入 onnxruntime 库测试是否可以初始化运行时
    import onnxruntime as ort

    print("onnxruntime version:", ort.__version__)

    _model = ort.InferenceSession(enc_p_io.getvalue())
    _model.get_outputs()
    del _model
