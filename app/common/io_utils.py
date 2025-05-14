# coding:utf-8
import requests
import json


def download_file(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"文件下载保存成功：{save_path}")
            return True
        else:
            print(f"文件下载失败，状态码：{response.status_code} - {url}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"文件url请求异常：{e}")
        return False


def save_text_to_file(text_content, file_name):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(text_content)
        print(f"文件写入成功：{file_name}")
    except Exception as e:
        print(f"文件写入出错 - {file_name}：{e}")


def save_json_to_file(json_data, file_name):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)
        print(f"文件写入成功：{file_name}")
    except Exception as e:
        print(f"文件写入出错 - {file_name}：{e}")


def read_text_from_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            text_content = file.read()
        return text_content
    except FileNotFoundError:
        print(f"文件未找到：{file_name}")
        return None
    except Exception as e:
        print(f"文件读取出错 - {file_name}：{e}")
        return None
