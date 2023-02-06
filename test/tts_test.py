# -*- coding: utf-8 -*-
# @Time    : 12/19/22 10:03 PM
# @FileName: tts_test.py
# @Software: PyCharm
# @Github    ：sudoskys

from event import TTS_Generate, TTS_REQ

dicts = {
    "model_name": "1374_epochs.pth",
    "task_id": 2,
    "text": "[ZH]你好[ZH]",
    "speaker_id": 0
}
tts_req = TTS_REQ(**dicts)
_model_path = f"../model/{tts_req.model_name}"
_reqTTS = TTS_Generate(model_path=_model_path)
_continue, _msg = _reqTTS.load_model()
if _continue:
    _result = _reqTTS.convert(text=tts_req.text,
                              task_id=tts_req.task_id,
                              speaker_ids=tts_req.speaker_id)
    print(_result)
else:
    print(_msg)
