# coding:utf-8
import os
import sys
import argparse
import warnings
import numpy as np
import json
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy import optimize
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pyannote.audio import Inference, Model

sys.path.append("/home/rise/qiutzh/service-audio2text-sd/app/open_toolkit")
from speakerlab.utils.config import Config
from speakerlab.utils.builder import build
from speakerlab.utils.utils import merge_vad, silent_print, download_model_from_modelscope, circle_pad
from speakerlab.utils.fileio import load_audio

# from app.open_toolkit.speakerlab.utils.config import Config
# from app.open_toolkit.speakerlab.utils.builder import build
# from app.open_toolkit.speakerlab.utils.utils import merge_vad, silent_print, download_model_from_modelscope, circle_pad
# from app.open_toolkit.speakerlab.utils.fileio import load_audio

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")


# sd
def get_speaker_embedding_model(device: torch.device = None, cache_dir: str = None):
    conf = {
        'model_id': 'iic/speech_campplus_sv_zh_en_16k-common_advanced',
        'revision': 'v1.0.0',
        'model_ckpt': 'campplus_cn_en_common.pt',
        'embedding_model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },  # 类和参数
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
            },  # 类和参数
        }
    }

    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    pretrained_model_path = os.path.join(cache_dir, conf['model_ckpt'])
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model, feature_extractor


def get_cluster_backend():
    conf = {
        'cluster': {
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args': {
                'cluster_type': 'spectral',
                'mer_cos': 0.8,
                'min_num_spks': 1,
                'max_num_spks': 15,
                'min_cluster_size': 4,
                'oracle_num': None,
                'pval': 0.012,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)


def get_voice_activity_detection_model(device: torch.device = None, cache_dir: str = None):
    conf = {
        'model_id': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        'revision': 'v2.0.4',
    }
    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=cache_dir,
            device='cpu' if device is None else '%s:%s' % (device.type, device.index) if device.index else device.type,
            disable_pbar=True,
            disable_update=True,
        )
    return vad_pipeline


def get_segmentation_model(use_auth_token, device: torch.device = None, cache_dir: str = None):
    segmentation_params = {
        'segmentation': 'pyannote/segmentation-3.0',
        'segmentation_batch_size': 32,
        'use_auth_token': use_auth_token,
    }
    model = Model.from_pretrained(
        segmentation_params['segmentation'],
        use_auth_token=segmentation_params['use_auth_token'],
        strict=False,
        cache_dir=cache_dir
    )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params['segmentation_batch_size'],
        device=device,
    )
    return segmentation


# asr
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


def download_model():
    model_cache_dir = "/home/rise/pretrained_file"
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # embedding_model, feature_extractor = get_speaker_embedding_model(device, model_cache_dir)
    # vad_model = get_voice_activity_detection_model(device, model_cache_dir)
    # cluster = get_cluster_backend()
    #
    # hf_access_token = ""
    # segmentation_model = get_segmentation_model(hf_access_token, device, f"{model_cache_dir}/pyannote")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr_pipeline = get_asr_model(device, model_cache_dir)
    print("Done!")


if __name__ == '__main__':
    download_model()
