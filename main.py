# -*- coding: utf-8 -*-
# @Time    : 12/19/22 9:09 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from celery_worker import tts_order

from api_server import TTS_REQ

app = FastAPI()


@app.post("/tts/generate")
async def tts(tts_req: TTS_REQ):
    return tts_order.delay(tts_req.dict())


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=9557, reload=True, log_level="debug", workers=1)
