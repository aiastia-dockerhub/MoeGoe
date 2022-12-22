# Server

CPU 推理。没有用到 GPU.

Based on Fastapi. Only tts and only support some type model(lazy me?)

Server Main Body Include:

```
server.py  # main
api_server.py # lib
celery_worker.py #闲置
tts_test.py # test
```

```
Server 部分在不并入上游前，以 GPL V2协议开源。
```

Install Req
`pip install -r requirements.txt`

`apt install libsndfile1`

Mkdir `model` and run `server.py` to start this server.

After that,Fastapi Docs:http://127.0.0.1:9557/docs

## Config

Server requirements for model placement

```
model----
 ---- somemodel.pth
 ---- somemodel.pth.json (yep,config.json)
 ---- info.json
```

**info.json**
use for perload
`{"model":["somemodel.pth"]}`

## Param

**POST**

```python
from pydantic import BaseModel


class TTS_REQ(BaseModel):
    model_name: str = ""
    task_id: int = 1
    text: str = "[ZH]你好[ZH]"
    speaker_id: int = 0
    audio_type: str = "ogg"  # flac wav ogg

```

**RETURN**

```python
from pydantic import BaseModel


class TTS_REQ_DATA(BaseModel):
    code: int = 404
    msg: str = "unknown error"
    audio: str = ""
    speaker: str = ""
    model_type: str = ""
```

## Other

- Other Api implementations https://github.com/fumiama/MoeGoe Just found out after writing, SAD

# Links_

- [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
- [Pretrained models](https://github.com/CjangCjengh/TTSModels)

# How to use

Run MoeGoe.exe

```
Path of a VITS model: path\to\model.pth
Path of a config file: path\to\config.json
INFO:root:Loaded checkpoint 'path\to\model.pth' (iteration XXX)
```

## Text to speech

```
TTS or VC? (t/v):t
Text to read: こんにちは。
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Speaker ID: 0
Path to save: path\to\demo.wav
Successfully saved!
```

## Voice conversion

```
TTS or VC? (t/v):v
Path of an audio file to convert:
path\to\origin.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Original speaker ID: 0
Target speaker ID: 6
Path to save: path\to\demo.wav
Successfully saved!
```

## HuBERT-VITS

```
Path of a hubert-soft model: path\to\hubert-soft.pt
Path of an audio file to convert:
path\to\origin.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Target speaker ID: 6
Path to save: path\to\demo.wav
Successfully saved!
```

## W2V2-VITS

```
Path of a w2v2 dimensional emotion model: path\to\model.onnx
TTS or VC? (t/v):t
Text to read: こんにちは。
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Speaker ID: 0
Path of an emotion reference: path\to\reference.wav
Path to save: path\to\demo.wav
Successfully saved!
```
