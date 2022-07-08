# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import numbers
import os.path as osp

from essmc2.utils.file_systems import FS
from .registry import TRANSFORMS
from ..utils.video_reader.frame_sampler import do_frame_sample
from ..utils.video_reader.video_reader import VideoReaderWrapper


@TRANSFORMS.register_class()
class LoadVideoFromFile(object):
    """ Open video file, extract frames, convert to tensor.

    Args:
        num_frames (int): T dimension value.
        sample_type (str): See
            `from essmc2.utils.video_reader import FRAME_SAMPLERS; print(FRAME_SAMPLERS)` to get candidates,
            default is 'interval'.
        clip_duration (Optional[float]): Needed for 'interval' sampling type.
        decoder (str): Video decoder name, default is decord.

    """

    def __init__(self,
                 num_frames,
                 sample_type='interval',
                 clip_duration=None,
                 decoder='decord'):

        try:
            import decord
        except:
            import warnings
            warnings.warn(f"You may run `pip install decord==0.6.0` to use {self.__class__.__name__}")
            exit(-1)

        self.num_frames = num_frames
        self.sample_type = sample_type
        self.clip_duration = clip_duration
        assert self.sample_type in ('uniform', 'interval', 'segment'), \
            f'Expected sample type in (uniform, interval, segment), got {self.sample_type}'
        if self.sample_type == 'interval':
            assert isinstance(self.clip_duration, numbers.Number), \
                "Interval style sampling needs clip_duration not None"
        self.decoder = decoder

    def __call__(self, item):
        """
        Args:
            item (dict):
                item['meta']['prefix'] (Optional[str]): Prefix of video_path.
                item['meta']['video_path'] (str): Required.
                item['meta']['clip_id'] (Optional[int]): Multi-view test needs it, default is 0.
                item['meta']['num_clips'] (Optional[int]): Multi-view test needs it, default is 1.
                item['meta']['start_sec'] (Optional[float]): Uniform sampling needs it.
                item['meta']['end_sec'] (Optional[float]): Uniform sampling needs it.

        Returns:
            item(dict):
                item['video'] (torch.Tensor): a THWC tensor.
        """
        meta = item['meta']
        video_path = meta['video_path'] if 'prefix' not in meta else osp.join(meta['prefix'], meta['video_path'])

        with FS.get_from(video_path) as local_path:
            vr = VideoReaderWrapper(local_path, decoder=self.decoder)

            params = dict()
            clip_id = meta.get('clip_id') or 0
            num_clips = meta.get('num_clips') or 1
            if self.sample_type == 'interval':
                # default is test mode for interval and segment
                params.update(clip_duration=self.clip_duration, clip_id=clip_id, num_clips=num_clips)
            elif self.sample_type == "segment":
                # default is test mode for interval and segment
                params.update(clip_id=clip_id, num_clips=num_clips)
            else:
                # uniform, needs start_sec, clip_duration or end_sec
                start_sec = meta['start_sec']
                if 'end_sec' in meta:
                    end_sec = meta['end_sec']
                elif self.clip_duration is not None:
                    end_sec = start_sec + self.clip_duration
                else:
                    raise ValueError("Uniform sampling needs start_sec & end_sec / start_sec & clip_duration")
                params.update(start_sec=start_sec, end_sec=end_sec)

            decode_list = do_frame_sample(self.sample_type, vr.len, vr.fps, self.num_frames, **params)
            item['video'] = vr.sample_frames(decode_list)

        return item
