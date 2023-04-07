# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 下午9:42
# @Author  : sudoskys
# @File    : event.py
# @Software: PyCharm
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal

# import librosa
# import numpy as np
import scipy
import soundfile as sf
import torch
from graiax import silkcoder
from pydantic import BaseModel

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence


# 类型
class VitsModelType(Enum):
    TTS = "tts"
    W2V2 = "w2v2"
    HUBERT_SOFT = "hubert-soft"


class TtsSchema(BaseModel):
    model_id: str = ""
    text: str = "我看它的意思是... 今日は晴天で、日差しがまぶしいです。新鮮な空気が流れています。今天的天气真好！"
    speaker_id: int = 0
    audio_type: Literal["ogg", "wav", "flac", "silk"] = "wav"
    length_scale: float = 1
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    sample_rate: int = 22050
    load_prefer: bool = True


class ParseText(object):
    def __init__(self):
        pass

    @staticmethod
    def get_label_value(text, label, default, warning_name='value'):
        try:
            value = re.search(rf'\[{label}=(.+?)\]', text)
            if value:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            else:
                value = default
        except ValueError as e:
            print(f'Invalid {warning_name}!', e)
            value = default
        return value, text

    @staticmethod
    def get_label(text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    @staticmethod
    def get_text(text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_stn_tst(self, c_text, hps_ms_config):
        _cleaned, c_text = self.get_label(c_text, 'CLEANED')
        _stn_tst = self.get_text(c_text, hps_ms_config, cleaned=_cleaned)
        return _stn_tst

    def clean_text(self, c_text):
        _, c_text = self.get_label_value(c_text, 'LENGTH', 0, warning_name='length scale')
        _, c_text = self.get_label_value(c_text, 'NOISE', 0, warning_name='noise scale')
        _, c_text = self.get_label_value(c_text, 'NOISEW', 0, warning_name='deviation of noise')
        _, c_text = self.get_label(c_text, 'CLEANED')
        return c_text

    def parse(self,
              c_text: str,
              length: float = 1,
              noise: float = 0.667,
              noise_w: float = 0.8,
              ):
        _length_scale, c_text = self.get_label_value(c_text, 'LENGTH', length, warning_name='length scale')
        _noise_scale, c_text = self.get_label_value(c_text, 'NOISE', noise, warning_name='noise scale')
        _noise_scale_w, c_text = self.get_label_value(c_text, 'NOISEW', noise_w, warning_name='deviation of noise')
        return _length_scale, _noise_scale, _noise_scale_w


class TtsGenerate(object):
    def __init__(self, model_path: str, model_config_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.model_config_path = model_config_path if model_config_path else f"{model_path}.json"
        self.device = device

        self._out_path = f"./tts/{0}.wav"

        self.hps_ms_config = utils.get_hparams_from_file(self.model_config_path)
        self.net_g_ms = self.load_model()

        self.n_speakers, self.n_symbols, self.emotion_embedding, self.speakers, self.use_f0 = self.parse_hps_ms(
            hps_ms=self.hps_ms_config)

        if self.n_symbols != 0:
            if not self.emotion_embedding:
                self.model_type = VitsModelType.TTS
            else:
                self.model_type = VitsModelType.W2V2
        else:
            self.model_type = VitsModelType.HUBERT_SOFT

    def load_model(self):
        # 判定是否存在模型
        if not Path(self.model_path).exists() or not Path(self.model_config_path).exists():
            return None
        model = SynthesizerTrn(
            len(self.hps_ms_config.symbols),
            self.hps_ms_config.data.filter_length // 2 + 1,
            self.hps_ms_config.train.segment_size // self.hps_ms_config.data.hop_length,
            n_speakers=self.hps_ms_config.data.n_speakers,
            **self.hps_ms_config.model)
        utils.load_checkpoint(self.model_path, model)
        model.eval().to(torch.device(self.device))
        return model

    @property
    def usable(self):
        return self.net_g_ms is not None

    def get_model_info(self):
        model_info = self.hps_ms_config.info if 'info' in self.hps_ms_config.keys() else {
            "name": None,
            "description": None,
            "author": None,
            "cover": None,
            "email": None
        }
        return model_info

    @staticmethod
    def load_prefer_noise(hps_ms, noise_scale, noise_scale_w, length_scale):
        # 生成默认噪声
        infer_prefer = hps_ms.infer if 'infer' in hps_ms.keys() else None
        if infer_prefer is not None:
            if 'noise_scale' in infer_prefer.keys():
                noise_scale = infer_prefer.noise_scale
            if 'noise_scale_w' in infer_prefer.keys():
                noise_scale_w = infer_prefer.noise_scale_w
            if 'length_scale' in infer_prefer.keys():
                length_scale = infer_prefer.length_scale
        return noise_scale, noise_scale_w, length_scale

    @staticmethod
    def parse_hps_ms(hps_ms):
        # 角色
        _n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        # 符号
        _n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
        _emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
        # 角色列表
        _speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
        _use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
        return _n_speakers, _n_symbols, _emotion_embedding, _speakers, _use_f0

    def get_speaker_list(self):
        # 二维数组 [id,name]
        id_list = []
        for ids, name in enumerate(self.speakers):
            id_list.append({"id": ids, "name": name})
        return id_list

    def find_speaker(self, speaker_ids: int):
        # 确定 ID
        _msg = "ok"
        find = False
        speaker_name = "none"
        speaker_list = self.get_speaker_list()
        for item in speaker_list:
            if speaker_ids == item["id"]:
                speaker_name = item['name']
                find = True
        if not find:
            # speaker_ids = speaker_list[0]["id"]
            speaker_name = speaker_list[0]["name"]
            _msg = "Not Find Speaker,Use 0"
        return _msg, speaker_name

    def infer(self,
              c_text: str,
              speaker_ids: int = 0,
              audio_type: str = Literal["ogg", "wav", "flac", "silk"],
              length_scale: float = 1,
              noise_scale: float = 0.667,
              noise_scale_w: float = 0.8,
              sample_rate: int = None,
              load_prefer: bool = True,
              ):
        """
        :param c_text: 文本
        :param speaker_ids: 角色ID，0为默认
        :param audio_type: 音频类型，ogg/wav/flac/silk
        :param length_scale: 长度缩放，决定了音频时长
        :param noise_scale: 噪声缩放，决定了音频噪声
        :param noise_scale_w: 噪声缩放宽度
        :param sample_rate: 采样率
        :param load_prefer: 是否加载模型偏好
        :return:
        """
        c_text = ParseText().clean_text(c_text)
        _stn_tst = ParseText().get_stn_tst(c_text, self.hps_ms_config)

        # 读模型偏好
        if load_prefer:
            noise_scale, noise_scale_w, length_scale = self.load_prefer_noise(self.hps_ms_config,
                                                                              noise_scale,
                                                                              noise_scale_w,
                                                                              length_scale
                                                                              )
        # 规则化文本覆盖
        # length_scale, noise_scale, noise_scale_w = ParseText().parse(c_text,
        #                                                             length=length_scale,
        #                                                             noise=noise_scale,
        #                                                             noise_w=noise_scale_w)
        # 构造对应 tensor
        with torch.no_grad():
            _x_tst = _stn_tst.unsqueeze(0)
            _x_tst_lengths = torch.LongTensor([_stn_tst.size(0)])
            _sid = torch.LongTensor([speaker_ids])
            _audio = self.net_g_ms.infer(_x_tst, _x_tst_lengths, sid=_sid,
                                         noise_scale=noise_scale,
                                         noise_scale_w=noise_scale_w,
                                         length_scale=1.0 / length_scale)[0][0, 0].data.cpu().float().numpy()
        # 写出返回
        _file = BytesIO()
        sample_rate = self.hps_ms_config.data.sampling_rate if not sample_rate else sample_rate
        sample_rate = int(sample_rate)
        sample_rate = 24000 if sample_rate < 0 else sample_rate
        # 使用 scipy 将 Numpy 数据写入字节流
        if audio_type == "ogg":
            sf.write(_file, _audio, sample_rate, format='ogg', subtype='vorbis')
        elif audio_type == "wav":
            # Write out audio as 24bit PCM WAV
            sf.write(_file, _audio, sample_rate, format='wav', subtype='PCM_24')
        elif audio_type == "flac":
            # Write out audio as 24bit Flac
            sf.write(_file, _audio, sample_rate, format='flac', subtype='PCM_24')
        elif audio_type == "silk":
            # Write out audio as 24bit Flac
            _file = BytesIO(initial_bytes=silkcoder.encode(_audio))
        else:
            scipy.io.wavfile.write(_file, sample_rate, _audio)
        _file.seek(0)
        # 获取 wav 数据
        return _file
