# -*- coding: utf-8 -*-
# @Time    : 12/19/22 9:06 PM
# @FileName: celery_worker.py.py
# @Software: PyCharm
# @Github    ：sudoskys
# !/usr/bin/python3
# !--*-- coding: utf-8 --*--
# !/usr/bin/python3
# !--*-- coding: utf-8 --*--
import time

from celery import Celery
from celery.utils.log import get_task_logger

from api_server import TTS_Generate, TTS_REQ

# 实例化 Celery
celery = Celery('tasks', broker='amqp://localhost//')

# 创建 logger，以显示日志信息
celery_log = get_task_logger(__name__)


# 创建任务函数，以订单(Order) 为例，异步进行
@celery.task
def tts_order(tts_req):
    _reqTTS = TTS_Generate(model_path=tts_req.get("model_name"))
    _continue, _msg = _reqTTS.load_model()
    if not _continue:
        return _msg
    return _reqTTS.convert(text=tts_req.get("text"),
                           task_id=tts_req.get("task_id"),
                           speaker_ids=tts_req.get("speaker_id"))
