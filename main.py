# -*- coding: utf-8 -*-
# @Time    : 2023/4/8 下午4:02
# @Author  : sudoskys
# @File    : main.py.py
# @Software: PyCharm
import os

import uvicorn
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel


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
