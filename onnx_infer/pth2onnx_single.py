import torch

import onnx_models_single
from infer import commons
import utils
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("config.json")
symbols = hps.symbols
net_g = ONNXVITS_models_single.SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()
_ = utils.load_checkpoint("model.pth", net_g)

text1 = get_text("watashiha.", hps)
stn_tst = text1
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    sid = torch.tensor([0])
    o = net_g(x_tst, x_tst_lengths, sid=sid)
