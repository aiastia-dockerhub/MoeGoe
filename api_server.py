# -*- coding: utf-8 -*-
# @Time    : 12/19/22 10:59 AM
# @FileName: apiserver.py
# @Software: PyCharm
# @Github    ：sudoskys
#
import re
import base64
# 初始化文件流
from io import BytesIO
#
import utils
import commons
from models import SynthesizerTrn
from torch import no_grad, LongTensor
#
from text import text_to_sequence
#
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
#
# import uvicorn
# from fastapi import FastAPI, Depends, status, HTTPException
#
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

#
import scipy
from io import BytesIO


#
# from mel_processing import spectrogram_torch
# from text import _clean_text


# 类型
class MODEL_TYPE(Enum):
    TTS = "tts"
    W2V2 = "w2v2"
    HUBERT_SOFT = "hubert-soft"


class Utils(object):
    @staticmethod
    def get_text(text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    @staticmethod
    def get_speakers(speakers, escape=False):
        if len(speakers) > 100:
            return
        _speaker = []
        for ids, name in enumerate(speakers):
            _speaker.append([ids, name])
        return _speaker

    @staticmethod
    def get_label_value(text, label, default, warning_name='value'):
        value = re.search(rf'\[{label}=(.+?)\]', text)
        if value:
            try:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            except Exception as e:
                print(f'Invalid {warning_name}!', e)
        else:
            value = default
        return value, text

    @staticmethod
    def get_label(text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    @staticmethod
    def deal_hps_ms(hps_ms):
        # 角色
        _n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        # 符号
        _n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
        _emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
        #
        _speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
        _use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
        return _n_speakers, _n_symbols, _emotion_embedding, _speakers, _use_f0

    @staticmethod
    def get_model_config_path(model_path):
        _model = model_path
        _config = f"{model_path}.json"
        return _model, _config

    @staticmethod
    def get_net(model_path):
        _model, _config = Utils.get_model_config_path(model_path=model_path)
        # 读取配置文件
        _hps_ms = utils.get_hparams_from_file(_config)
        _n_speakers, _n_symbols, _emotion_embedding, _speakers, _use_f0 = Utils.deal_hps_ms(hps_ms=_hps_ms)
        # 载入模型
        if _n_symbols != 0:
            if not _emotion_embedding:
                model_type = MODEL_TYPE.TTS
            else:
                model_type = MODEL_TYPE.W2V2
        else:
            model_type = MODEL_TYPE.HUBERT_SOFT
        _net_g_ms = SynthesizerTrn(
            _n_symbols,
            _hps_ms.data.filter_length // 2 + 1,
            _hps_ms.train.segment_size // _hps_ms.data.hop_length,
            n_speakers=_n_speakers,
            emotion_embedding=_emotion_embedding,
            **_hps_ms.model)
        return _net_g_ms, _hps_ms, model_type,


class TTS_Generate(object):
    def __init__(self, model_path: str):
        self._out_path = f"./tts/{0}.wav"
        self.obj = None
        self.model_type = None
        self.hps_ms = None
        self.net_g_ms = None
        self.use_f0 = None
        self.speakers = None
        self.emotion_embedding = None
        self.n_symbols = None
        self.n_speakers = None
        self.model_path = model_path

    def load_model(self):
        _model, _config = Utils.get_model_config_path(model_path=self.model_path)
        if not Path(_model).exists() or not Path(_config).exists():
            return False, TTS_REQ_DATA(code=404, msg=f"CANT FIND MODEL", audio="").dict()
        _net_g_ms, _hps_ms, model_type = Utils.get_net(model_path=self.model_path)
        self.n_speakers, self.n_symbols, self.emotion_embedding, self.speakers, self.use_f0 = Utils.deal_hps_ms(
            hps_ms=_hps_ms)
        self.net_g_ms = _net_g_ms
        self.hps_ms = _hps_ms
        self.model_type = model_type
        _ = self.net_g_ms.eval()
        self.obj = utils.load_checkpoint(self.model_path, self.net_g_ms)
        return True, TTS_REQ_DATA(code=200, msg=f"YOU SHOULD NOT SEE THIS MSG", audio="").dict()

    @staticmethod
    def no_found(msg):
        return TTS_REQ_DATA(code=404, msg=f"{msg}:unknown or unsupported type TTS", audio="").dict()

    def get_speaker_list(self):
        # 二维数组 [id,name]
        id_list = []
        if len(self.speakers) > 100:
            print("TOO MANY SPEAKERS")
            return id_list.append({"id": 0, "name": "default"})
        for ids, name in enumerate(self.speakers):
            id_list.append({"id": ids, "name": name})
        return id_list

    def ordinary(self,
                 c_text: str,
                 speaker_ids: int = 0,
                 length: float = 1,
                 noise: float = 0.667,
                 noise_w: float = 0.8):
        _msg = "ok"
        # 的链式调用
        _length_scale, c_text = Utils.get_label_value(c_text, 'LENGTH', length, 'length scale')
        _noise_scale, c_text = Utils.get_label_value(c_text, 'NOISE', noise, 'noise scale')
        _noise_scale_w, c_text = Utils.get_label_value(c_text, 'NOISEW', noise_w, 'deviation of noise')
        _cleaned, c_text = Utils.get_label(c_text, 'CLEANED')
        _stn_tst = Utils.get_text(c_text, self.hps_ms, cleaned=_cleaned)

        # 确定 ID
        find = False
        speaker_name = "none"
        speaker_list = self.get_speaker_list()
        for item in speaker_list:
            if speaker_ids == item["id"]:
                speaker_name = item['name']
                find = True
        if not find:
            speaker_ids = speaker_list[0]["id"]
            speaker_name = speaker_list[0]["name"]
            _msg = "Not Find Speaker,Use 0"

        # 构造对应 tensor
        with no_grad():
            _x_tst = _stn_tst.unsqueeze(0)
            _x_tst_lengths = LongTensor([_stn_tst.size(0)])
            _sid = LongTensor([speaker_ids])
            _audio = self.net_g_ms.infer(_x_tst,
                                         _x_tst_lengths,
                                         sid=_sid,
                                         noise_scale=_noise_scale,
                                         noise_scale_w=_noise_scale_w,
                                         length_scale=_length_scale)[0][0, 0].data.cpu().float().numpy()
        # 写出返回

        # 首先将 NumPy 数据转换为 WAV 数据
        # wavData = scipy.io.wavfile.write(self._out_path, self.hps_ms.data.sampling_rate, _audio)
        # wav_data = Path(self._out_path).open("rb").read()

        file = BytesIO()
        # 使用 scipy 将 Numpy 数据写入字节流
        scipy.io.wavfile.write(file, self.hps_ms.data.sampling_rate, _audio)
        # 获取 wav 数据
        wav_data = file.getvalue()
        audio_base64 = base64.b64encode(wav_data)
        return TTS_REQ_DATA(code=200,
                            msg=_msg,
                            audio=audio_base64,
                            speaker=speaker_name,
                            model_type="tts").dict()

    def convert(self, text, task_id: int = 1, speaker_ids: int = 0):
        """
        逻辑调度类
        """
        self._out_path = f"./tts/{task_id}.wav"
        _type = self.model_type
        if _type == MODEL_TYPE.TTS:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids)
            return _res

        elif _type == MODEL_TYPE.HUBERT_SOFT:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids)
            return _res

        elif _type == MODEL_TYPE.W2V2:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids)
            return _res

        else:
            _res = self.no_found(msg=f"Unknown type {_type}")
            return _res


class TTS_REQ(BaseModel):
    model_name: str = ""
    task_id: int = 1
    text: str = "[ZH]你好[ZH]"
    speaker_id: int = 0


class TTS_REQ_DATA(BaseModel):
    code: int = 404
    msg: str = "unknown error"
    audio: str = ""
    speaker: str = ""
    model_type: str = ""
