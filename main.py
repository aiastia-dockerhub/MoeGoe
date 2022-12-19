# -*- coding: utf-8 -*-
# @Time    : 12/19/22 9:09 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import uvicorn
from loguru import logger
from fastapi import BackgroundTasks, FastAPI

# from celery_worker import tts_order

from api_server import TTS_REQ, TTS_Generate

app = FastAPI()

# 日志机器
logger.add(sink='run.log',
           format="{time} - {level} - {message}",
           level="INFO",
           rotation="500 MB",
           enqueue=True)


@app.post("/tts/generate")
def tts(tts_req: TTS_REQ):
    _model_path = f"./model/{tts_req.model_name}"
    _reqTTS = TTS_Generate(model_path=_model_path)
    _continue, _msg = _reqTTS.load_model()
    if not _continue:
        return _msg
    try:
        _result = _reqTTS.convert(text=tts_req.text,
                                  task_id=tts_req.task_id,
                                  speaker_ids=tts_req.speaker_id)
    except Exception as e:
        logger.error(e)
        return {"code": -1, "msg": "Error!"}
    else:
        return _result

    # tts_order.delay(tts_req.dict())


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=9557, reload=True, log_level="debug", workers=1)
