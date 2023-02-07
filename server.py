# -*- coding: utf-8 -*-
# @Time    : 12/19/22 9:09 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import json

import rtoml
import uvicorn
from loguru import logger
from fastapi import FastAPI
# from celery_worker import tts_order
from event import TTS_REQ, TTS_Generate

app = FastAPI()
with open("model/index.json", "r", encoding="utf-8") as f:
    models_info = json.load(f)
_Model_list = {model_name: TTS_Generate(model_path=f"./model/{model_name}") for model_name in
               models_info["model"]}

# 日志机器
logger.add(sink='run.log',
           format="{time} - {level} - {message}",
           level="INFO",
           rotation="500 MB",
           enqueue=True)


@app.post("/tts/generate")
def tts(tts_req: TTS_REQ):
    global _Model_list
    _reqTTS = _Model_list.get(tts_req.model_name)
    _reqTTS: TTS_Generate
    if not _reqTTS:
        return {"code": -1, "msg": "Not Found!"}
    _continue, _msg = _reqTTS.check_model()
    if not _continue:
        return _msg
    try:
        _result = _reqTTS.convert(text=tts_req.text,
                                  task_id=tts_req.task_id,
                                  speaker_ids=tts_req.speaker_id,
                                  audio_type=tts_req.audio_type,
                                  )
    except Exception as e:
        logger.error(e)
        return {"code": -1, "msg": "Error!"}
    else:
        return _result
    # tts_order.delay(tts_req.dict())


# Run
CONF = rtoml.load(open("config.toml", 'r'))
ServerConf = CONF.get("server") if CONF.get("server") else {}
AutoReload = ServerConf.get("reload") if ServerConf.get("reload") else False
ServerHost = ServerConf.get("host") if ServerConf.get("host") else "127.0.0.1"
ServerPort = ServerConf.get("port") if ServerConf.get("port") else 9557

if __name__ == '__main__':
    uvicorn.run('server:app', host=ServerHost, port=ServerPort, reload=AutoReload, log_level="debug", workers=1)
