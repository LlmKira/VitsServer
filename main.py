# -*- coding: utf-8 -*-
# @Time    : 2023/4/8 下午4:02
# @Author  : sudoskys
# @File    : main.py.py
# @Software: PyCharm
import rtoml
import uvicorn
from loguru import logger
from pydantic import BaseModel

# Run
CONF = rtoml.load(open("config.toml", 'r'))


class FastApiConf(BaseModel):
    reload: bool = False
    host: str = "127.0.0.1"
    port: int = 0
    workers: int = 1


ServerConf = CONF.get("server") if CONF.get("server") else {}
FastApi = FastApiConf(**ServerConf)

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
