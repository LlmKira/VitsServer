import os
from json import loads

import dotenv
import librosa
import torch
from loguru import logger
from numpy import float32
from torch import load, FloatTensor


def get_device(by_torch: bool = True):
    dotenv.load_dotenv()
    if torch.cuda.is_available():
        logger.info("GPU Is Available!")
        infer_device = "gpu"
        if by_torch:
            infer_device = "cuda"
        only_cpu = os.environ.get('VITS_DISABLE_GPU', False) == 'true'
        if only_cpu:
            infer_device = "cpu"
    else:
        infer_device = "cpu"
    return infer_device


DEVICE = get_device()


class HParams(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = loads(data)

    hparams = HParams(**config)
    return hparams


def load_audio_to_torch(full_path, target_sampling_rate):
    audio, sampling_rate = librosa.load(full_path, sr=target_sampling_rate, mono=True)
    return FloatTensor(audio.astype(float32))
