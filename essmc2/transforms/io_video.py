# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp
import random

import torch

from essmc2.utils.file_systems import FS
from .registry import TRANSFORMS


def _interval_based_sampling(vid_length, vid_fps, target_fps, clip_idx, num_clips, num_frames, interval,
                             minus_interval=False):
    """ Generates the frame index list using interval based sampling.

    Args:
        vid_length (int): The length of the whole video (valid selection range).
        vid_fps (float): The original video fps.
        target_fps (int): The target decode fps.
        clip_idx (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video.
        num_clips (int): The total clips to be sampled from each video.
            Combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
        num_frames (int): Number of frames in each sampled clips.
        interval (int): The interval to sample each frame.
        minus_interval (bool):

    Returns:
        index (torch.Tensor): The sampled frame indexes.
    """
    if num_frames == 1:
        index = [random.randint(0, vid_length - 1)]
    else:
        # transform FPS
        clip_length = num_frames * interval * vid_fps / target_fps

        max_idx = max(vid_length - clip_length, 0)
        if clip_idx == -1:  # random sampling
            start_idx = random.uniform(0, max_idx)
        else:
            if num_clips == 1:
                start_idx = max_idx / 2
            else:
                start_idx = max_idx * clip_idx / num_clips
        if minus_interval:
            end_idx = start_idx + clip_length - interval
        else:
            end_idx = start_idx + clip_length - 1

        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, vid_length - 1).long()

    return index


def _segment_based_sampling(vid_length, clip_idx, num_clips, num_frames, random_sample):
    """ Generates the frame index list using segment based sampling.

    Args:
        vid_length (int): The length of the whole video (valid selection range).
        clip_idx (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video.
        num_clips (int): The total clips to be sampled from each video.
            Combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
        num_frames (int): Number of frames in each sampled clips.
        random_sample (bool): Whether or not to randomly sample from each segment. True for train and False for test.

    Returns:
        index (torch.Tensor): The sampled frame indexes.
    """
    index = torch.zeros(num_frames)
    index_range = torch.linspace(0, vid_length, num_frames + 1)
    for idx in range(num_frames):
        if random_sample:
            index[idx] = random.uniform(index_range[idx], index_range[idx + 1])
        else:
            if num_clips == 1:
                index[idx] = (index_range[idx] + index_range[idx + 1]) / 2
            else:
                index[idx] = index_range[idx] + (index_range[idx + 1] - index_range[idx]) * (clip_idx + 1) / num_clips
    index = torch.round(torch.clamp(index, 0, vid_length - 1)).long()

    return index


@TRANSFORMS.register_class()
class DecodeVideoToTensor(object):
    def __init__(self,
                 num_frames,
                 target_fps=30,
                 sample_mode='interval',
                 sample_interval=4,
                 sample_minus_interval=False,
                 repeat=1):
        """ DecodeVideoToTensor
        Args:
            num_frames (int): Decode frames number.
            target_fps (int): Decode frames fps, default is 30.
            sample_mode (str): Interval or segment sampling, default is interval.
            sample_interval (int): Sample interval between output frames for interval sample mode, default is 4.
            sample_minus_interval (bool): If minus interval for interval sample mode, default is False.
            repeat (int): Number of clips to be decoded from each video, if repeat > 1, outputs will be named like
                'video-0', 'video-1'. Normally, 1 for classification task, 2 for contrastive learning.
        """

        try:
            import decord
        except:
            import warnings
            warnings.warn(f"You may run `pip install decord==0.6.0` to use {self.__class__.__name__}")
            exit(-1)

        self.num_frames = num_frames
        self.target_fps = target_fps
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.sample_minus_interval = sample_minus_interval
        self.repeat = repeat

    def __call__(self, item):
        """ Call to invoke decode
        Args:
            item (dict): A dict contains which file to decode and how to decode.
                Normally, it has structure like
                    {
                        "meta": {
                            "prefix" (str, None): if not None, prefix will be added to video_path.
                            "video_path" (str): Absolute (prefix is None) or relative path.
                            "clip_idx" (int): -1 means random sampling, >=0 means do temporal crop.
                            "num_clips" (int): if clip_idx >= 0, clip_idx must < num_clips
                        }
                    }

        Returns:
            A dict contains original input item and "video" tensor.
        """
        import decord
        decord.bridge.set_bridge('torch')

        meta = item["meta"]
        video_path = meta["video_path"] \
            if "prefix" not in meta else osp.join(meta["prefix"], meta["video_path"])

        with FS.get_from(video_path) as local_path:
            vr = decord.VideoReader(local_path)
            # default is test mode
            clip_id = meta.get("clip_id") or 0
            num_clips = meta.get("num_clips") or 1

            vid_len = len(vr)
            vid_fps = vr.get_avg_fps()

            frame_list = []
            for _ in range(self.repeat):
                if self.sample_mode == "interval":
                    decode_list = _interval_based_sampling(vid_len, vid_fps, self.target_fps, clip_id, num_clips,
                                                           self.num_frames, self.sample_interval,
                                                           self.sample_minus_interval)
                else:
                    decode_list = _segment_based_sampling(vid_len, clip_id, num_clips, self.num_frames, clip_id == -1)

                # Decord gives inconsistent result for avi files. Getting full frames will fix it, although slower.
                # See https://github.com/dmlc/decord/issues/195
                if video_path.lower().endswith('avi'):
                    full_decode_list = list(range(0, torch.max(decode_list).item() + 1))
                    full_frames = vr.get_batch(full_decode_list)
                    frames = full_frames[decode_list].clone()
                else:
                    frames = vr.get_batch(decode_list).clone()

                frame_list.append(frames)

            if self.repeat == 1:
                item["video"] = frame_list[0]
            else:
                for idx, frame_tensor in zip(range(self.repeat), frame_list):
                    item[f"video-{idx}"] = frame_tensor

            del vr
        return item
