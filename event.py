# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 下午9:42
# @Author  : sudoskys
# @File    : event.py
# @Software: PyCharm
import io
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal, List, Optional, Union

import librosa
import numpy as np
# import librosa
# import numpy as np
import soundfile as sf
import torch
from graiax import silkcoder
from loguru import logger
from pydantic import BaseModel

import utils
from component.warp import Parse
from onnx_infer.infer import commons
from onnx_infer.utils.onnx_utils import RunONNX
from pth2onnx import VitsExtractor
from text import text_to_sequence


# 类型
class VitsModelType(Enum):
    TTS = "vits"
    W2V2 = "w2v2"
    HUBERT_SOFT = "soft-vits-vc"


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


class ConvertSchema(BaseModel):
    model_id: str = ""
    # audio: BytesIO = None
    speaker_id: int = 0
    audio_type: Literal["ogg", "wav", "flac", "silk"] = "wav"
    length_scale: float = 1
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    sample_rate: int = 22050
    load_prefer: bool = True


class InferTask(BaseModel):
    infer_sample: str
    speaker_ids: int = 0
    audio_type: Literal["ogg", "wav", "flac", "silk"] = "wav"
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    sample_rate: int = None


class ConvertTask(InferTask):
    infer_sample: BytesIO
    fn0: float = 0.1

    class Config:
        arbitrary_types_allowed = True


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
        # text_norm = np.array(text_norm, dtype=np.int64)
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
    """
    批次语音合成技术
    """

    def __init__(self,
                 model_config_path: str,
                 model_path: str = None,
                 device: str = "cpu",
                 load_prefer: bool = True,
                 # hubert_soft_model_path=None,
                 ):
        self.load_prefer = load_prefer
        self.model_config_path = model_config_path
        self.model_path = model_path if model_path else None
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
        # load hubert-soft model
        self.hubert = None
        if self.n_symbols == 0:
            self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)

    def load_model(self):
        # 判定是否存在模型
        if not Path(self.model_config_path).exists():
            return None
        try:
            _vits_base = VitsExtractor().warp_pth(model_config_path=self.model_config_path, model_path=self.model_path)
        except Exception as e:
            logger.exception(
                f"Model Not Found Or Convert Error: {e} {self.model_config_path} ，可能是模型格式不正确，或者模型文件字段缺失")
            return None
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if utils.DEVICE == "cpu":
            providers = ['CPUExecutionProvider']
        model = RunONNX(model=_vits_base, providers=providers)
        # model = onnx_infer.SynthesizerTrn(
        #     len(self.hps_ms_config.symbols),
        #     self.hps_ms_config.data.filter_length // 2 + 1,
        #     self.hps_ms_config.train.segment_size // self.hps_ms_config.data.hop_length,
        #     onnx_model=_vits_base,
        #     n_speakers=self.hps_ms_config.data.n_speakers,
        #     **self.hps_ms_config.model
        # )
        # utils.load_checkpoint(self.model_path, model, None)
        # model.eval()
        # utils.load_checkpoint(self.model_path, model)
        # model.eval().to(torch.device(self.device))
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

    def infer_soft_vc(self,
                      task: ConvertTask = None,
                      ):
        """
        语音转换任务推理接口
        :param task 任务
        :return:
        """
        if not self.hubert:
            raise Exception("hubert-soft model not found")
        # 读模型偏好
        if self.load_prefer:
            task.noise_scale, task.noise_scale_w, task.length_scale = self.load_prefer_noise(self.hps_ms_config,
                                                                                             task.noise_scale,
                                                                                             task.noise_scale_w,
                                                                                             task.length_scale
                                                                                             )
        # 给RAW音频数据加上格式
        # 载入并采样
        # TODO librosa 采样问题，我们无法采样不标准的音频
        _file = sf.SoundFile(task.infer_sample, "r", samplerate=44100, channels=1, subtype="PCM_16")
        # audio, sampling_rate = sf.read(_file, dtype='float32', always_2d=True)
        audio, sampling_rate = librosa.load(_file, sr=16000, mono=True)
        # audio16000, sampling_rate = librosa.load(path=audio, sr=16000, mono=True)
        if self.use_f0:
            # audio, sampling_rate = sf.read(_file, samplerate=self.hps_ms_config.data.sampling_rate)
            audio, sampling_rate = librosa.load(_file, sr=self.hps_ms_config.data.sampling_rate, mono=True)
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        # 计算时长
        duration = audio.shape[0] / sampling_rate
        if duration > 30:
            raise Exception("audio too long")
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        # 创建 units
        with torch.inference_mode():
            _x_tst = self.hubert.units(torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(utils.DEVICE))

        with torch.no_grad():
            _x_tst_lengths = np.array([_x_tst.shape[1]], dtype=np.int64)  # torch.LongTensor([_stn_tst.size(0)])
            _sid = np.array([task.speaker_ids], dtype=np.int64)
            scales = np.array([task.noise_scale, task.noise_scale_w, 1.0 / task.length_scale], dtype=np.float32)
            scales.resize((1, 3))
            ort_inputs = {
                'input': _x_tst,
                'input_lengths': _x_tst_lengths,
                'scales': scales,
                'sid': _sid
            }
            audio = np.squeeze(self.net_g_ms.run(model_input=ort_inputs))
            audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
            audio = np.clip(audio, -32767.0, 32767.0)
            _audio = audio.astype(np.int16)
        del _x_tst, _x_tst_lengths, _sid
        return audio

    def infer_vits(self,
                   task: InferTask = None,
                   ):
        """
        文本转语音任务推理接口
        :param task 任务
        :return:
        """
        task.infer_sample = ParseText().clean_text(task.infer_sample)
        _stn_tst = ParseText().get_stn_tst(task.infer_sample, self.hps_ms_config)

        # 读模型偏好
        if self.load_prefer:
            task.noise_scale, task.noise_scale_w, task.length_scale = self.load_prefer_noise(self.hps_ms_config,
                                                                                             task.noise_scale,
                                                                                             task.noise_scale_w,
                                                                                             task.length_scale
                                                                                             )
        # 规则化文本覆盖
        # length_scale, noise_scale, noise_scale_w = ParseText().parse(c_text,
        #                                                             length=length_scale,
        #                                                             noise=noise_scale,
        #                                                             noise_w=noise_scale_w)
        # 构造对应 tensor
        with torch.no_grad():
            _x_tst = _stn_tst.unsqueeze(0).numpy()
            _x_tst_lengths = np.array([_x_tst.shape[1]], dtype=np.int64)  # torch.LongTensor([_stn_tst.size(0)])
            _sid = np.array([task.speaker_ids], dtype=np.int64)
            scales = np.array([task.noise_scale, task.noise_scale_w, 1.0 / task.length_scale], dtype=np.float32)
            scales.resize((1, 3))
            """
            _x_tst = _stn_tst[np.newaxis, :].astype(np.float32)
            _x_tst_lengths = np.array([_x_tst.shape[1]], dtype=np.int64)
            _sid = np.array([task.speaker_ids], dtype=np.int64)
            scales = np.array([task.noise_scale, task.noise_scale_w, 1.0 / task.length_scale], dtype=np.float32)
            scales.resize(1, 3)
            _x_tst = _x_tst.astype(np.int64)
            _x_tst_lengths = _x_tst_lengths.astype(np.int64)
            """
            ort_inputs = {
                'input': _x_tst,
                'input_lengths': _x_tst_lengths,
                'scales': scales,
                'sid': _sid
            }
            audio = np.squeeze(self.net_g_ms.run(model_input=ort_inputs))
            audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
            audio = np.clip(audio, -32767.0, 32767.0)
            _audio = audio.astype(np.int16)
        # 释放内存
        del _stn_tst, _x_tst, _x_tst_lengths, _sid
        return _audio

    def encode_audio(self, audio, sample_rate, audio_type):
        # 写出返回
        _file = BytesIO()
        sample_rate = self.hps_ms_config.data.sampling_rate if not sample_rate else sample_rate
        sample_rate = int(sample_rate)
        sample_rate = 24000 if sample_rate < 0 else sample_rate

        if audio_type == "ogg":
            sf.write(_file, audio, sample_rate, format='ogg', subtype='vorbis')
        elif audio_type == "wav":
            # Write out audio as 24bit PCM WAV
            sf.write(_file, audio, sample_rate, format='wav', subtype='PCM_24')
        elif audio_type == "flac":
            # Write out audio as 24bit Flac
            sf.write(_file, audio, sample_rate, format='flac', subtype='PCM_24')
        elif audio_type == "silk":
            # Write out audio as 24bit Flac
            byte_io = io.BytesIO(bytes())
            sf.write(byte_io, audio, sample_rate)
            _file = BytesIO(initial_bytes=silkcoder.encode(byte_io))
            del byte_io
        else:
            sf.write(_file, audio, sample_rate, format='wav', subtype='PCM_24')
        _file.seek(0)
        return _file

    def infer_task(self,
                   task: Union[InferTask, ConvertTask] = None,
                   ):
        """
        :param task 任务
        :return:
        """
        if isinstance(task, ConvertTask):
            _audio = self.infer_soft_vc(task=task)
        else:
            _audio = self.infer_vits(task=task)
        _file = self.encode_audio(_audio, task.sample_rate, task.audio_type)
        # 获取 wav 数据
        return _file

    def infer_task_bat(self, task_list: List[Union[InferTask, ConvertTask]]):
        """
        :param task_list 任务列表
        :return:
        """
        # 检查任务列表，确定编码类型和采样率一样
        for task in task_list:
            if task.sample_rate != task_list[0].sample_rate:
                raise Exception("sample_rate must be same")
            if task.audio_type != task_list[0].audio_type:
                raise Exception("audio_type must be same")
        # 批量推理
        _file = []
        for task in task_list:
            if isinstance(task, InferTask):
                _audio = self.infer_vits(task=task)
                _file.append(_audio)
            if isinstance(task, ConvertTask):
                _audio = self.infer_soft_vc(task=task)
                _file.append(_audio)
        # 合并音频
        audio_data = np.concatenate(_file, axis=0)
        return self.encode_audio(audio_data, task_list[0].sample_rate, task_list[0].audio_type)

    def create_vits_task(self,
                         c_text: str,
                         speaker_ids: int = 0,
                         audio_type: str = "wav",
                         length_scale: float = 1.0, noise_scale: float = 0.667, noise_scale_w: float = 0.8,
                         sample_rate: Optional[int] = None
                         ) -> List[InferTask]:
        """
        :param c_text 语句
        :param speaker_ids 说话人id
        :param noise_scale 音频噪声
        :param noise_scale_w 音频噪声权重
        :param length_scale 音频长度
        :param sample_rate 采样率
        :param audio_type 音频类型
        :return:
        """
        if self.hubert:
            logger.warning("hubert is not None, maybe you shouldn't use `create_vits_task` function")
        _task_list = []
        parse = Parse()
        sentence_cell = parse.create_cell(c_text, merge_same=False, cell_limit=140)
        sentence_task = parse.pack_up_task(sentence_cell=sentence_cell, task_limit=140, strip=True)
        for sentence in sentence_task:
            last = InferTask(
                infer_sample="".join(sentence),
                speaker_ids=speaker_ids,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sample_rate=sample_rate,
                audio_type=audio_type
            )
            _task_list.append(last)
        # TEST
        # test = last.copy()
        # test.infer_sample = "[ZH]测试,多任务正常工作[ZH]"
        # _task_list.append(test)
        return _task_list
