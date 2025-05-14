# coding:utf-8
import os
import json
import argparse
import re
import torch
import torch.distributed as dist
import sys
import jieba
import logging
import modelscope
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.asr.paraformer.infer_sd import SpeakerDiarizationInference

sys.path.append("/home/rise/qiutzh/service-audio2text-sd/app/open_toolkit")
from speakerlab.utils.utils import silent_print, download_model_from_modelscope

logger = logging.getLogger('jieba')
logger.setLevel(logging.CRITICAL)
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'


def get_asr_model(device: str = None, cache_dir: str = None):
    conf = {
        'model_id': 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'model_revision': 'v2.0.4',
    }
    cache_dir = download_model_from_modelscope(conf['model_id'], conf['model_revision'], cache_dir)
    # Initialize the ASR pipeline
    # asr_pipeline = pipeline(
    #     task=Tasks.auto_speech_recognition,
    #     model=conf['model_id'],
    #     model_revision=conf['model_revision'],
    #     device=device,
    #     disable_pbar=True,
    #     disable_update=True,
    #     cache_dir=cache_dir
    # )
    asr_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=cache_dir,
        device=device,
        disable_pbar=True,
        disable_update=True,
        cache_dir=cache_dir
    )
    return asr_pipeline


class ASRInferenceParaformer(object):
    def __init__(self):
        self.out_dir = "/home/rise/qiutzh/service-audio2text-sd/exp_infer_asr"  # 输出文件夹
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        self.checkpoint_dir = "/home/rise/pretrained_file"  # 模型本地统一缓存路径
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.asr_pipeline = get_asr_model(self.device, self.checkpoint_dir)

    def get_trans_sentence(self, wav_path):
        """
        Performs ASR on the given audio file and aligns the text with timestamps.

        Args:
            wav_path (str): Path to the audio file.

        Returns:
            list: A list of sentences with text and timestamps.
        """
        sentence_info = [[]]
        punc_pattern = r'[,.!?;:"\-—…、，。！？；：“”‘’]'
        asr_result = self.asr_pipeline(wav_path, return_raw_text=True)[0]  # todo
        raw_text = asr_result['raw_text']
        text = asr_result['text']
        timestamp = asr_result['timestamp']
        timestamp = [[i[0] / 1000, i[1] / 1000] for i in timestamp]
        raw_text_list = raw_text.split()
        assert len(timestamp) == len(raw_text_list)
        text_pt = 0
        for i, wd in enumerate(raw_text_list):  # 文本和时间戳对齐
            cache_text = ''
            while text_pt < len(text) and cache_text.lower().replace(' ', '') != wd.lower():
                cache_text += text[text_pt]
                text_pt += 1
            if text_pt > len(text):
                print('[ERROR]: The ASR results may have wrong format, skip processing %s' % wav_path)
                return []
            while text_pt < len(text) and (text[text_pt] == ' ' or text[text_pt] in punc_pattern):
                cache_text += text[text_pt]
                text_pt += 1
            sentence_info[-1].append([cache_text, timestamp[i]])
            if cache_text[-1] in punc_pattern and text_pt < len(text):
                sentence_info.append([])
        return sentence_info

    def match_spk(self, sentence, output_field_labels):
        """
        Matches the given sentence with speaker IDs based on timestamps.对分段语音匹配说话者id

        Args:
            sentence (list): A list of words with timestamps.
            output_field_labels (list): A list of speaker segments with start and end times.

        Returns:
            list: A list of speaker IDs matched with the sentence.
        """
        # sentence: [[word1, timestamps1], [word2, timestamps2], ...]
        # output_field_labels: [[st, ed, spkid], ...]
        if len(sentence) == 0:
            return []
        st_sent = sentence[0][1][0]
        ed_sent = sentence[-1][1][1]
        max_overlap = 0
        overlap_per_spk = {}
        for st_spk, ed_spk, spk in output_field_labels:
            overlap_dur = min(ed_sent, ed_spk) - max(st_sent, st_spk)
            if spk not in overlap_per_spk:
                overlap_per_spk[spk] = 0
            if overlap_dur > 0:
                overlap_per_spk[spk] += overlap_dur
        overlap_per_spk_list = [[spk, overlap_per_spk[spk]] for spk in overlap_per_spk if overlap_per_spk[spk] > 0]
        # sort by duration
        overlap_per_spk_list = sorted(overlap_per_spk_list, key=lambda x: x[1], reverse=True)
        overlap_per_spk_list = [i[0] for i in overlap_per_spk_list]
        return overlap_per_spk_list

    def distribute_spk(self, sentence_info, output_field_labels):
        """
        Distributes speaker IDs to each word in the sentence based on timestamps.

        Args:
            sentence_info (list): A list of sentences with text and timestamps.
            output_field_labels (list): A list of speaker segments with start and end times.

        Returns:
            list: A list of words with text, timestamps, and speaker IDs.
        """
        #  sentence_info: [[[word1, timestamps1], [word2, timestamps2], ...], ...]
        last_spk = 0
        for sentence in sentence_info:
            main_spks = self.match_spk(sentence, output_field_labels)
            main_spk = main_spks[0] if len(main_spks) > 0 else last_spk
            for i, wd in enumerate(sentence):
                wd_spks = self.match_spk([wd], output_field_labels)
                if main_spk in wd_spks:
                    sentence[i].append(main_spk)
                elif len(wd_spks) > 0:
                    sentence[i].append(wd_spks[0])
                else:
                    sentence[i].append(last_spk)
            last_spk = sentence[-1][2]
        if len(sentence_info) == 0:
            return []
        #  sentence_info_with_spk: [text_string, timeinterval, spk]
        sentence_info = [j for i in sentence_info for j in i]
        sentence_info_with_spk_merge = [sentence_info[0]]
        punc_pattern = r'[,.!?;:"\-—…、，。！？；：“”‘’]'
        for i in sentence_info[1:]:
            if i[2] == sentence_info_with_spk_merge[-1][2] and \
                    i[1][0] < sentence_info_with_spk_merge[-1][1][1] + 2:
                sentence_info_with_spk_merge[-1][0] += i[0]
                sentence_info_with_spk_merge[-1][1][1] = i[1][1]
            else:
                sentence_info_with_spk_merge.append(i)
        return sentence_info_with_spk_merge

    def infer_demo(self, audio_file, output_field_labels):
        sentence_info = self.get_trans_sentence(audio_file)
        if not sentence_info:
            return []
        return self.distribute_spk(sentence_info, output_field_labels)

    def infer(self, audio_file):
        ans = []
        print(f"输入语音：{audio_file}")
        # 1. 首先做说话者分离
        sd_infer_model = SpeakerDiarizationInference()
        dia_result = sd_infer_model.infer(audio_file)
        print(f'说话者分离结果：\n{dia_result}')
        # 挨个处理每一份
        audio_ids = [os.path.splitext(os.path.basename(list(item.keys())[0]))[0] for item in dia_result]
        for i, audio_dia_info in enumerate(dia_result):
            audio_id = audio_ids[i]
            output_field_labels = []
            for dia_list in audio_dia_info.values():
                for v in dia_list:  # [start, end, speaker_id]
                    output_field_labels.append([float(v[0]), float(v[1]), v[2]])
                    # output_field_labels.append([float(v[0]), float(v[0]) + float(v[1]), v[2]])

            # 2. 然后做语音识别，需包含时间戳信息
            sentence_info = self.get_trans_sentence(audio_file)

            # 3. 后处理，收集不同说话者文本
            if not sentence_info:
                return []
            sentence_info_with_spk = self.distribute_spk(sentence_info, output_field_labels)
            ans = sentence_info_with_spk
            print(f"语音识别结果：\n{ans}")
            # 数据写入
            output_trans_path = os.path.join(self.out_dir, audio_id + '.txt')
            with open(output_trans_path, 'w') as f:
                for text_string, timeinterval, spk in sentence_info_with_spk:
                    f.write('%s: [%.3f %.3f] %s\n' % (spk, timeinterval[0], timeinterval[1], text_string))
            msg = 'Transcripts of %s have been finished in %s' % (audio_id, self.out_dir)
            print(f'[INFO]: {msg}')
        return ans


def main():
    # audio_files = ["/home/rise/qiutzh/service-audio2text-sd/data/examples/2speakers_example.wav"]
    audio_files = [
        # "/home/rise/qiutzh/service-audio2text-tools/data/audio_tempdata_0418/35ce90cc-0b75-11ef-84d9-4f9fb6367409.wav",
        # "/home/rise/qiutzh/service-audio2text-tools/data/audio_tempdata_0418/4b2568d4-ab93-11ee-9b5d-b30f2649e394.wav",
        "/home/rise/qiutzh/service-audio2text-tools/data/audio_tempdata_0418/b6b4bb28-d153-11ee-a758-d71e97cf9d82.wav",
    ]
    asr_infer = ASRInferenceParaformer()
    for audio_file in audio_files:
        # result = asr_infer.infer_demo(audio_file, output_field_labels=[])
        result = asr_infer.infer(audio_file)
        print(result)


if __name__ == '__main__':
    main()
