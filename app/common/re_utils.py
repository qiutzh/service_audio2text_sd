# coding:utf-8
import re


def extract_uid_from_url(url):
    """
    从给定的 URL 中提取 uid 字段。

    :param url: 包含 uid 字段的 URL
    :return: 提取的 uid 字段，如果未找到则返回 None
    """
    # 使用正则表达式匹配 uid 字段
    match = re.search(r'uid=([^&]+)', url)
    if match:
        return match.group(1)
    else:
        return None
