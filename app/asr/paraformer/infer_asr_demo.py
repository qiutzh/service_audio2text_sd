# coding:utf-8
# coding:utf-8
import os
import json
import argparse
import re
import torch
import torch.distributed as dist
import sys

sys.path.append("/home/rise/qiutzh/service-audio2text-sd/app/open_toolkit")
from speakerlab.utils.utils import silent_print

import jieba
import logging

logger = logging.getLogger('jieba')
logger.setLevel(logging.CRITICAL)
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'

try:
    import modelscope
except ImportError:
    raise ImportError("Package \"modelscope\" not found. Please install them first.")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ASRInference:
    def __init__(self, asr_model_config, device=None):
        """
        Initializes the ASRInference class.

        Args:
            asr_model_config (dict): Configuration for the ASR model.
                It should contain 'model_id' and 'model_revision'.
            device (str, optional): Device to use for inference. Defaults to None.
        """
        self.asr_model_config = asr_model_config
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize the ASR pipeline
        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=self.asr_model_config['model_id'],
            model_revision=self.asr_model_config['model_revision'],
            device=self.device,
            disable_pbar=True,
            disable_update=True,
        )

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
        asr_result = self.asr_pipeline(wav_path, return_raw_text=True)[0]
        raw_text = asr_result['raw_text']
        text = asr_result['text']
        timestamp = asr_result['timestamp']
        timestamp = [[i[0] / 1000, i[1] / 1000] for i in timestamp]
        raw_text_list = raw_text.split()
        assert len(timestamp) == len(raw_text_list)
        text_pt = 0
        for i, wd in enumerate(raw_text_list):
            cache_text = ''
            while text_pt < len(text) and cache_text.lower().replace(' ', '') != wd.lower():
                cache_text += text[text_pt]
                text_pt += 1
            if text_pt > len(text):
                print(f'[ERROR]: The ASR results may have wrong format, skip processing {wav_path}')
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
        Matches the given sentence with speaker IDs based on timestamps.

        Args:
            sentence (list): A list of words with timestamps.
            output_field_labels (list): A list of speaker segments with start and end times.

        Returns:
            list: A list of speaker IDs matched with the sentence.
        """
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
        if len(sentence_info) == 0:
            return []
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

    def infer(self, audio_file, output_field_labels):
        """
        Performs ASR on the given audio file and combines the results with speaker diarization.

        Args:
            audio_file (str): Path to the audio file.
            output_field_labels (list): A list of speaker segments with start and end times.

        Returns:
            list: A list of words with text, timestamps, and speaker IDs.
        """
        sentence_info = self.get_trans_sentence(audio_file)
        if not sentence_info:
            return []
        return self.distribute_spk(sentence_info, output_field_labels)


def main():
    audio_files = ["/home/rise/qiutzh/service-audio2text-sd/data/examples/2speakers_example.wav"]
    gpu_id = 0
    if gpu_id < torch.cuda.device_count():
        device = 'cuda:%d' % gpu_id
    else:
        print(f"[WARNING]: Gpu {gpu_id} is not available. Use cpu instead.")
        device = 'cpu'

    asr_model_config = {
        'model_id': 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'model_revision': 'v2.0.4',
    }

    asr_infer = ASRInference(asr_model_config, device=device)
    output_field_labels = []  # Placeholder for speaker diarization results
    for audio_file in audio_files:
        result = asr_infer.infer(audio_file, output_field_labels)
        print(result)


if __name__ == '__main__':
    main()
