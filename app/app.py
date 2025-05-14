# coding: utf-8
import os
import time
import torch
import traceback
from fastapi import FastAPI
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.asr.paraformer.infer_asr import ASRInferenceParaformer
from app.common.io_utils import download_file, read_text_from_file
from app.common.re_utils import extract_uid_from_url
from app.common.audio_utils import process_audio_to_wav

app = FastAPI(docs_url=None, redoc_url=None)
app.openapi_version = "3.0.0"
if os.path.exists('static'):
    app.mount('/static', StaticFiles(directory='static'), name='static file')
else:
    app.mount('/static', StaticFiles(directory='app/static'), name='static file')


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url='static/swagger-ui-bundle.js',
        swagger_css_url='static/swagger-ui.css',
    )


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Input(BaseModel):
    audio_url: str = ""


class ASRResponse(BaseModel):
    code: int
    message: str
    data: str
    time_cost: float


asr_infer = ASRInferenceParaformer()


@app.post('/speech_recognizer_file', response_model=ASRResponse)
async def speech_recognizer_file(file: UploadFile = File(...)):
    # async def audio2text_openapi(files: List[UploadFile] = File(...)):
    # def audio2text_openapi(file: UploadFile = File(...)):
    result = []
    start_time = time.time()
    status_code = 200
    message = "SUCCESS"
    # temp_path = "/home/rise/qiutzh/service-audio2text-tools/data/audio_tempdata"
    temp_path = "/tmp/audio_tempdata"  # 记得将路径更改为实际存在的地址
    try:
        # 检查文件类型
        if not file.filename.endswith((".wav", ".mp3", ".ogg")):
            raise HTTPException(status_code=400, detail="不支持的文件格式，仅支持 WAV、MP3、OGG 格式")
        # 读取音频
        contents = await file.read()
        file_name = file.filename.split("/")[-1]
        file_path = f"{temp_path}/{file_name}"
        # 将音频下载到本地 - 直接下载mp3，和转化wav结果不一样
        with open(file_path, "wb") as f:
            f.write(contents)
        # 转录音频
        resp = asr_infer.infer(file_path)
        if isinstance(resp, list):
            for l in resp:
                json_data = {"sentence": l[0],
                             "start": l[1][0],
                             "end": l[1][1],
                             "speaker": l[2]}
                result.append(json_data)
        else:
            result = resp
            pass
    except Exception as err:
        status_code = 500
        message = f"ASR接口调用异常:\n{traceback.format_exc()}"
        print(message)
        # raise HTTPException(status_code=500, detail="语音转录失败")
    finally:
        # # 清理临时文件
        # file_name = file.filename.split("/")[-1]
        # file_path = f"{temp_path}/{file_name}"
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass
    # return {"code": status_code,
    #         "message": message,
    #         "data": result,
    #         "time_cost": time.time() - start_time}
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": message,
            "data": result,
            "time_cost": time.time() - start_time
        }
    )


@app.post('/speech_recognizer', response_model=ASRResponse)
async def speech_recognizer(input: Input):
    result = []
    start_time = time.time()
    status_code = 200
    message = "SUCCESS"
    temp_path = "/tmp/audio_tempdata/"  # 音频保存地址
    try:
        # 下载语音文件到本地
        url = input.audio_url
        uid_str = extract_uid_from_url(url)  # 获取录音标识id
        record_file_name = f"{uid_str}.wav"
        audio_save_path = temp_path + record_file_name
        # download_file(url, audio_save_path)
        process_audio_to_wav(url, audio_save_path)
        # 转录音频
        resp = asr_infer.infer(audio_save_path)
        if isinstance(resp, list):
            for l in resp:
                json_data = {"sentence": l[0],
                             "start": l[1][0],
                             "end": l[1][1],
                             "speaker": l[2]}
                result.append(json_data)
        else:
            result = resp
            pass
    except Exception as err:
        status_code = 500
        message = f"ASR接口调用异常:\n{traceback.format_exc()}"
        print(message)
        # raise HTTPException(status_code=500, detail="语音转录失败")
    finally:
        # # 清理临时文件
        # file_name = file.filename.split("/")[-1]
        # file_path = f"{temp_path}/{file_name}"
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": message,
            "data": result,
            "time_cost": time.time() - start_time
        }
    )
