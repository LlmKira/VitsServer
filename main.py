# -*- coding: utf-8 -*-
# @Time    : 2023/4/8 下午4:02
# @Author  : sudoskys
# @File    : main.py.py
# @Software: PyCharm
import os
import pathlib

import requests
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm


def download_file(folder_path, file_name, url, max_retries=3):
    # 拼接
    file_path = os.path.join(folder_path, file_name)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if not chunk:
                    break
                file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()
        if os.path.getsize(file_path) == total_size:
            logger.success(f"初始化模型下载成功: {file_path}")
            return True
        else:
            os.remove(file_path)
            if max_retries > 0:
                return download_file(folder_path, file_name, url, max_retries - 1)
    return False


# Run

class FastApiConf(BaseModel):
    reload: bool = False
    host: str = "127.0.0.1"
    port: int = 9557
    workers: int = 1


# Load environment variables from .env file
load_dotenv()

host = str(os.environ.get('VITS_SERVER_HOST', "0.0.0.0"))
port = int(os.environ.get('VITS_SERVER_PORT', 9557))
reload = os.environ.get('VITS_SERVER_RELOAD', False) == 'true'
workers = int(os.environ.get('VITS_SERVER_WORKERS', 1))
FastApi = FastApiConf(reload=reload, host=host, port=port, workers=workers)

init_model = os.environ.get('VITS_SERVER_INIT_MODEL', None)
init_config = os.environ.get('VITS_SERVER_INIT_CONFIG', None)

# 查询是否存在init模型在路径下
if not pathlib.Path("model/init.json").exists() and init_config:
    download_file("model", "init.json", init_config)

# 获得文件链接的文件后缀
if init_model:
    file_name = os.path.basename(init_model)
    file_ext = os.path.splitext(file_name)[-1]
    if not pathlib.Path(f"model/init{file_ext}").exists():
        download_file("model", f"init{file_ext}", init_model)

if FastApi.reload:
    logger.warning("reload 参数有内容修改自动重启服务器，启用可能导致连续重启导致 CPU 满载")

if __name__ == '__main__':
    uvicorn.run('server:app',
                host=FastApi.host,
                port=FastApi.port,
                reload=FastApi.reload,
                log_level="debug",
                workers=FastApi.workers
                )
