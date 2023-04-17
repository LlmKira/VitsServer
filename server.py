# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 下午11:25
# @Author  : sudoskys
# @File    : server.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 12/19/22 9:09 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import io
import pathlib
import shutil
from typing import Annotated

import psutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger

import utils
from component.nlp_utils.detect import DetectSentence
from component.warp import Parse
# from celery_worker import tts_order
from event import TtsGenerate, TtsSchema, InferTask, ConvertTask, ConvertSchema

# 日志机器
logger.add(sink='run.log',
           format="{time} - {level} - {message}",
           level="INFO",
           rotation="500 MB",
           enqueue=True)

app = FastAPI()

_Model_list = {}
pathlib.Path("./model").mkdir(parents=True, exist_ok=True)
for model_config_path in pathlib.Path("./model").iterdir():
    if model_config_path.is_file() and model_config_path.suffix == ".json":
        pth_model_path = model_config_path.parent / f'{model_config_path.stem}.pth'
        onnx_model_path = model_config_path.parent / f'{model_config_path.stem}.onnx'
        if pathlib.Path(pth_model_path).exists() or pathlib.Path(onnx_model_path).exists():
            _load_model = TtsGenerate(model_config_path=str(model_config_path.absolute()), device=utils.get_device())
            _Model_list[model_config_path.stem] = _load_model
            if _load_model.net_g_ms:
                logger.success(f"{model_config_path} 对应的模型配置加载成功")
            else:
                logger.warning(f"{model_config_path} 对应的模型配置加载失败")
        else:
            logger.warning(f"{model_config_path} 没有对应的模型文件")


# 主页
@app.get("/")
def index():
    # 获取当前内存剩余
    _rest = psutil.virtual_memory().percent
    return {"code": 0, "message": "success", "data": {"memory": _rest}}


# FastApi 获取模型列表和信息
@app.get("/model/list")
def tts_model(show_speaker: bool = False, show_ms_config: bool = False):
    global _Model_list
    _data = []
    # 构建模型信息
    for _model_name, _model in _Model_list.items():
        _model: TtsGenerate
        _item = {
            "model_id": _model_name,
            "model_info": _model.get_model_info(),
            # 是否某个属性是NONE  ,
            "model_class": _model.model_type.value,
        }
        if show_speaker:
            _item["speaker"] = _model.get_speaker_list()
            _item["speaker_num"]: _model.n_speakers
        if show_ms_config:
            _item["ms_config"] = _model.hps_ms_config
        _data.append(
            _item
        )
    return {"code": 0, "message": "success", "data": _data}


# 获取模型名称对应的设置参数
@app.get("/model/info")
def tts_model_info(model_id: str):
    global _Model_list
    server_build = _Model_list.get(model_id)
    server_build: TtsGenerate
    if not server_build:
        return {"code": -1, "message": "Not Found!", "data": {}}
    return {"code": 0, "message": "success", "data": server_build.hps_ms_config}


# 处理传入文本为Vits格式包装
@app.post("/tts/parse")
def tts_parse(text: str, strip: bool = False,
              merge_same: bool = False, cell_limit: int = 140,
              filter_space: bool = True):
    _result = {}
    try:
        parse = Parse()
        _merge = parse.create_cell(sentence=text, merge_same=merge_same, cell_limit=cell_limit,
                                   filter_space=filter_space)
        _result["detect_code"] = DetectSentence.detect_code(text)
        _result["parse"] = _merge
        _result["raw_text"] = text
        _result["result"] = parse.build_sentence(_merge, strip=strip)
    except Exception as e:
        logger.exception(e)
        # raise HTTPException(status_code=500, detail="Error When Process Text!")
        return {"code": -1, "message": "Error!", "data": {}}
    return {"code": 0, "message": "success", "data": _result}


@app.post("/svc/convert")
async def svc_convert(model_id: Annotated[str, Form()],
                      noise_scale: Annotated[float, Form()] = 0.667,
                      length_scale: Annotated[float, Form()] = 1.4,
                      audio_type: Annotated[str, Form()] = "wav",
                      speaker_id: Annotated[int, Form()] = 0, file: UploadFile = File(...)):
    global _Model_list
    tts_req = ConvertSchema()
    tts_req.model_id = model_id
    tts_req.noise_scale = noise_scale
    tts_req.length_scale = length_scale
    tts_req.audio_type = audio_type
    tts_req.speaker_id = speaker_id
    server_build = _Model_list.get(tts_req.model_id, None)
    server_build: TtsGenerate
    # 检查模型是否存在
    if not server_build:
        raise HTTPException(status_code=404, detail="Model Not Found!")
    if not server_build.hubert:
        raise HTTPException(status_code=404, detail="Hubert Model Not Found!")
    # 检查请求合法性
    if not file:
        raise HTTPException(status_code=400, detail="Text is Empty!")
    # if tts_req.audio_type not in TtsSchema().audio_type:
    #    raise HTTPException(status_code=400, detail="Audio Type is Invalid!")
    conv = io.BytesIO()
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(conv, buffer)
    _task = ConvertTask(infer_sample=conv,
                        speaker_ids=tts_req.speaker_id,
                        audio_type=tts_req.audio_type,
                        length_scale=tts_req.length_scale,
                        noise_scale=tts_req.noise_scale,
                        )
    # tts_order.delay(_task)
    _result = server_build.infer_task(_task)
    return StreamingResponse(_result, media_type="application/octet-stream")


@app.post("/tts/generate")
async def tts(tts_req: TtsSchema, auto_parse: bool = False):
    global _Model_list
    server_build = _Model_list.get(tts_req.model_id, None)
    server_build: TtsGenerate
    # 检查模型是否存在
    if not server_build:
        raise HTTPException(status_code=404, detail="Model Not Found!")

    # 检查模型
    # if server_build.hubert:
    #     raise HTTPException(status_code=400, detail="self.n_symbols==0 and Hubert Model Be Found!")

    # 检查请求合法性
    if not tts_req.text:
        raise HTTPException(status_code=400, detail="Text is Empty!")
    # if tts_req.audio_type not in TtsSchema().audio_type:
    #    raise HTTPException(status_code=400, detail="Audio Type is Invalid!")
    if auto_parse:
        _task = server_build.create_vits_task(c_text=tts_req.text,
                                              speaker_ids=tts_req.speaker_id,
                                              audio_type=tts_req.audio_type,
                                              length_scale=tts_req.length_scale,
                                              noise_scale=tts_req.noise_scale,
                                              noise_scale_w=tts_req.noise_scale_w
                                              )
    else:
        _task = [InferTask(infer_sample=tts_req.text,
                           speaker_ids=tts_req.speaker_id,
                           audio_type=tts_req.audio_type,
                           length_scale=tts_req.length_scale,
                           noise_scale=tts_req.noise_scale,
                           noise_scale_w=tts_req.noise_scale_w)]
    # 检查 speaker_id 合法性
    if tts_req.speaker_id >= server_build.n_speakers:
        raise HTTPException(status_code=400, detail="Speaker ID is Invalid!")
    try:
        server_build.load_prefer = tts_req.load_prefer
        _result = server_build.infer_task_bat(
            task_list=_task
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Error When Generate Voice!")
    else:
        _result.seek(0)
        return StreamingResponse(_result, media_type="application/octet-stream")
