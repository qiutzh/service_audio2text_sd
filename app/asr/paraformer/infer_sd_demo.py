# coding:utf-8
import os
import sys
import argparse
import warnings
import numpy as np
import json
import torch
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

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")


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
            },
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
            },
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


def get_segmentation_model(use_auth_token, device: torch.device = None):
    segmentation_params = {
        'segmentation': 'pyannote/segmentation-3.0',
        'segmentation_batch_size': 32,
        'use_auth_token': use_auth_token,
    }
    model = Model.from_pretrained(
        segmentation_params['segmentation'],
        use_auth_token=segmentation_params['use_auth_token'],
        strict=False,
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


class Diarization3Dspeaker():
    def __init__(self, device=None, include_overlap=False, hf_access_token=None, speaker_num=None,
                 model_cache_dir=None):
        if include_overlap and hf_access_token is None:
            raise ValueError("hf_access_token is required when include_overlap is True.")

        self.device = self.normalize_device(device)
        self.include_overlap = include_overlap

        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(self.device, model_cache_dir)
        self.vad_model = get_voice_activity_detection_model(self.device, model_cache_dir)
        self.cluster = get_cluster_backend()

        if include_overlap:
            self.segmentation_model = get_segmentation_model(hf_access_token, self.device)

        self.batchsize = 64
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None
        self.speaker_num = speaker_num

    def __call__(self, wav, wav_fs=None, speaker_num=None):
        wav_data = load_audio(wav, wav_fs, self.fs)

        vad_time = self.do_vad(wav_data)
        if self.include_overlap:
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = get_valid_field(count)
            vad_time = merge_vad(vad_time, valid_field)

        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]

        embeddings = self.do_emb_extraction(chunks, wav_data)

        speaker_num, output_field_labels = self.do_clustering(chunks, embeddings, speaker_num)

        if self.include_overlap:
            binary = self.post_process(output_field_labels, speaker_num, segmentations, count)
            timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
            output_field_labels = self.binary_to_segs(binary, timestamps)

        self.output_field_labels = output_field_labels
        return output_field_labels

    def do_vad(self, wav):
        vad_results = self.vad_model(wav[0])[0]
        vad_time = [[vad_t[0] / 1000, vad_t[1] / 1000] for vad_t in vad_results['value']]
        return vad_time

    def do_segmentation(self, wav):
        segmentations = self.segmentation_model({'waveform': wav, 'sample_rate': self.fs})
        frame_windows = self.segmentation_model.model.receptive_field

        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed, dur=1.5, step=0.75):
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        wavs = [wav[0, int(st * self.fs):int(ed * self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st: batch_st + self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    def do_clustering(self, chunks, embeddings, speaker_num=None):
        cluster_labels = self.cluster(
            embeddings,
            speaker_num=speaker_num if speaker_num is not None else self.speaker_num
        )
        speaker_num = cluster_labels.max() + 1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0] + frame_windows.duration / 2) \
                           :frame_windows.closest_frame(i[1] + frame_windows.duration / 2) \
                , i[2]] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            start_frame = frame_windows.closest_frame(c.start + frame_windows.duration / 2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j]) > 0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() \
                                             for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1] * speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                        data[:, r], align_chunk_cluster_frames[:, c]
                    )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1) == 0) & (cluster_frames.sum(-1) != 0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
        output_field_labels = []
        for k, k_scores in enumerate(binary.T):
            start = timestamps[0]
            is_active = k_scores[0] > threshold

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < threshold:
                        output_field_labels.append([round(start, 3), round(t, 3), k])
                        start = t
                        is_active = False
                else:
                    if y > threshold:
                        start = t
                        is_active = True

            if is_active:
                output_field_labels.append([round(start, 3), round(t, 3), k])
        return sorted(output_field_labels, key=lambda x: x[0])

    def save_diar_output(self, out_file, wav_id=None, output_field_labels=None):
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError('No results can be saved.')
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        wav_id = 'default' if wav_id is None else wav_id
        if out_file.endswith('rttm'):
            line_str = "SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, 'w') as f:
                for seg in output_field_labels:
                    seg_st, seg_ed, cluster_id = seg
                    f.write(line_str.format(wav_id, seg_st, seg_ed - seg_st, cluster_id))
        elif out_file.endswith('json'):
            out_json = {}
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                item = {
                    'start': seg_st,
                    'stop': seg_ed,
                    'speaker': cluster_id,
                }
                segid = wav_id + '_' + str(round(seg_st, 3)) + \
                        '_' + str(round(seg_ed, 3))
                out_json[segid] = item
            with open(out_file, mode='w') as f:
                json.dump(out_json, f, indent=2)
        else:
            raise ValueError('The supported output file formats are currently limited to RTTM and JSON.')

    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device


def get_valid_field(count):
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item() == 0 or i == len(count) - 1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field


def compressed_seg(seg_list):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1] + seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list


def init_model(include_overlap, hf_access_token):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dia_model = Diarization3Dspeaker(device, include_overlap, hf_access_token)
    return dia_model


def process_wav(wav_path, dia_model):
    try:
        dia_result = dia_model(wav_path)
        return {wav_path: dia_result}
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


class SpeakerDiarizationInference(object):
    def __init__(self,
                 hf_access_token="",
                 speaker_num=2,
                 include_overlap=False,
                 disable_pogress_bar=False,
                 nprocs=1):
        self.hf_access_token = hf_access_token
        self.speaker_num = speaker_num
        self.include_overlap = include_overlap
        self.disable_pogress_bar = disable_pogress_bar
        self.nprocs = nprocs

        self.out_type = "rttm"
        self.out_dir = "/home/rise/qiutzh/service-audio2text-sd/exp_infer"
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        # Initialize the diarization model
        self.dia_model = init_model(self.include_overlap, self.hf_access_token)

    def infer(self, audio_file):
        if audio_file.endswith('.wav'):
            wav_list = [audio_file]
        else:
            try:
                with open(audio_file, 'r') as f:
                    wav_list = [line.strip() for line in f.readlines()]
            except Exception as err:
                raise Exception('[ERROR]: Input should be a wav file or a wav list.')
        assert len(wav_list) > 0

        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=self.nprocs) as executor:
            futures = {executor.submit(process_wav, wav_path, self.dia_model): wav_path for wav_path in wav_list}

            results = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Merge results
        self.dia_result = results

        # Save results
        for res in results:
            if res:
                wav_path = list(res.keys())[0]
                wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
                if self.out_dir is not None:
                    out_file = os.path.join(self.out_dir, f"{wav_id}.{self.out_type}")
                else:
                    out_file = f"{wav_path.rsplit('.', 1)[0]}.{self.out_type}"
                self.dia_model.save_diar_output(out_file, wav_id, list(res.values())[0])

        print("[INFO]: Speaker Diarization inference finished.")
        return self.dia_result


def infer_demo():
    wav = "/home/rise/qiutzh/service-audio2text-sd/data/examples/2speakers_example.wav"
    sd_infer_model = SpeakerDiarizationInference()
    results = sd_infer_model.infer(wav)
    print(results)


if __name__ == '__main__':
    infer_demo()