# coding:utf-8
import requests
import traceback
from io import BytesIO
from pydub import AudioSegment


def convert_mp3_to_wav_in_memory(mp3_data):
    # 将 MP3 数据转换为音频段
    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3")
    # 转换为单声道并设置采样率
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    # 将音频段导出到内存中的字节流
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    # 获取 WAV 数据的字节
    wav_data = wav_buffer.getvalue()
    return wav_data


def process_audio_to_wav(url, dst_file):
    try:
        # 下载音频文件到内存
        response = requests.get(url)
        response.raise_for_status()
        # 将音频数据加载到内存中的字节流对象
        audio_data = BytesIO(response.content)
        try:
            audio = AudioSegment.from_file(audio_data, format="mp3")  # # 先尝试作为 MP3 文件打开
            if audio:
                audio = audio.set_channels(1)  # 转换为单声道并调整采样率
                audio = audio.set_frame_rate(16000)
                # 保存转换后的音频文件
                audio.export(dst_file, format="wav")
                print(f"MP3 音频转换为 WAV 成功，已保存到 {dst_file}")
        except Exception as mp3_err:  # 如果作为 MP3 打开失败，尝试作为 WAV 文件打开
            try:
                audio_data.seek(0)  # 重置字节流的位置
                audio = AudioSegment.from_file(audio_data, format="wav")
                if audio:  # 如果成功作为 WAV 打开，则直接保存
                    audio.export(dst_file, format="wav")
                    print(f"WAV 文件直接保存成功，已保存到 {dst_file}")
            except Exception as wav_err:
                print(f"无法识别的音频格式或打开失败: {wav_err}")
    except Exception as err:
        print(f"原始音频文件处理失败: {err}")
