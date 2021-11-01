# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import random

import torch
import torchvision.transforms._functional_video as F

from essmc2.transforms import TRANSFORMS, VideoTransform


@TRANSFORMS.register_class()
class MoSIGenerator(VideoTransform):
    """ Generator for pseudo camera motions with static masks in MoSI.

    See paper "Self-supervised Motion Learning from Static Images",
    Huang et al. 2021 (https://arxiv.org/abs/2104.00240) for details.

    The MoSI generator process goes as follows:
    (a) In the initialization stage, a `speed_set` is generated according
    to the config.
    (b) In the training stage, each speed in the `speed_set` is used to
    generate a sample from the given data.

    Args:
        crop_size (int, tuple): Crop window size from input video.
        num_frames (int): Output motion video frame numbers.
        num_speeds (int): Move speed to be built, for example, 5 in x axis means -2, -1, 0, 1, 2 x move speed.
        mode (str): Mode, train/eval/test.
        aspect_ratio (int, tuple): Output window aspect ratio.
        distance_jitter (int, tuple): Move distance jitter.
        data_mode (str): x/y/xy.
        decouple (bool): If true, move will only occur in ONE axis while the other will keep static.
        zero_out (bool): If true, static condition will not happen.
        static_mask_enable (bool): If true, a static mask will be covered on output videos.
        mask_size_ratio (tuple): Static mask size ratio.
        frame_size_standardize_enable (bool): If true, output video will be resized.
        standard_size (int, tuple): Resize params.
        transforms (List[dict], None):
        out_label_key (str):
    """

    def __init__(self,
                 crop_size,
                 num_frames,
                 num_speeds,
                 mode="test",
                 aspect_ratio=(1, 1),
                 distance_jitter=(1, 1),
                 data_mode="xy",
                 label_mode="joint",
                 decouple=True,
                 zero_out=False,
                 static_mask_enable=True,
                 mask_size_ratio=(0.3, 0.5),
                 frame_size_standardize_enable=True,
                 standard_size=320,
                 transforms=None,
                 out_label_key="mosi_label",
                 **kwargs
                 ):
        super(MoSIGenerator, self).__init__(**kwargs)
        self.out_label_key = out_label_key

        if isinstance(crop_size, tuple) or isinstance(crop_size, list):
            assert len(crop_size) <= 2
            self.crop_size = crop_size[0]
        else:
            self.crop_size = crop_size

        self.num_speeds = num_speeds
        self.distance_jitter = distance_jitter
        assert len(self.distance_jitter) == 2 and self.distance_jitter[0] <= self.distance_jitter[1]

        self.label_mode = label_mode
        self.num_frames = num_frames
        self.mode = mode
        self.static_mask_enable = static_mask_enable
        self.aspect_ratio = aspect_ratio
        self.mask_size_ratio = mask_size_ratio
        self.decouple = decouple
        self.data_mode = data_mode
        self.zero_out = zero_out
        self.frame_size_standardize_enable = frame_size_standardize_enable
        self.standard_size = standard_size

        self.transforms = TRANSFORMS.build(dict(type='Compose', transforms=transforms))

        self._initialize_speed_set()
        self.labels = self.label_generator()

    def _initialize_speed_set(self):
        """ Initialize speed set for x and y separately.
        Initialized speed set is a list of lists [speed_x, speed_y].

        First a set of all speeds are generated as `speed_all`, then
        the required speeds are taken from the `speed_all` according
        to the config on MoSI to generate the `speed_set` for MoSI.
        """

        # Generate all possible speed combinations
        self.speed_all = []
        _zero_included = False

        # for example, when the number of classes is defined as 5,
        # then the speed range for each axis is [-2, -1, 0, 1, 2]
        # negative numbers indicate movement in the negative direction
        self.speed_range = (
                torch.linspace(0, self.num_speeds - 1, self.num_speeds) - self.num_speeds // 2
        ).long()
        self.speed_min = int(min(self.speed_range))

        for x in self.speed_range:
            for y in self.speed_range:
                x, y = [int(x), int(y)]
                if x == 0 and y == 0:
                    if _zero_included:
                        continue
                    else:
                        _zero_included = True
                if self.decouple and x * y != 0:
                    # if decouple, then one of x,y must be 0
                    continue
                self.speed_all.append(
                    [x, y]
                )

        # select speeds from all speed combinations
        self.speed_set = []
        assert self.data_mode is not None
        if self.decouple:
            """
            Decouple means the movement is only on one axies. Therefore, at least one of
            the speed is zero.
            """
            if "x" in self.data_mode:
                for i in range(len(self.speed_all)):
                    if self.speed_all[i][0] != 0:  # speed of x is not 0
                        self.speed_set.append(self.speed_all[i])
            if "y" in self.data_mode:
                for i in range(len(self.speed_all)):
                    if self.speed_all[i][1] != 0:  # speed of y is not 0
                        self.speed_set.append(self.speed_all[i])
        else:
            # if not decoupled, all the speeds in the speed set is added in the speed set
            if "x" in self.data_mode and "y" in self.data_mode:
                self.speed_set = self.speed_all
            else:
                raise NotImplementedError(
                    "Not supported for data mode {} when DECOUPLE is set to true.".format(self.data_mode))

        if self.decouple and not self.zero_out:
            # add speed=0
            self.speed_set.append([0, 0])

    def sample_generator(self, video):
        out = []
        for speed_idx, speed in enumerate(self.speed_set):
            # generate all the samples according to the speed set
            num_input_frames, h, w, c = video.shape
            frame_idx = random.randint(0, num_input_frames - 1)
            selected_frame = video[frame_idx]  # H, W, C

            # standardize the frame size
            if self.frame_size_standardize_enable:
                selected_frame = self.frame_size_standardize(selected_frame)

            # generate the sample index
            h, w, c = selected_frame.shape
            speed_x, speed_y = speed
            start_x, end_x = self.get_crop_params(speed_x / (self.num_speeds // 2), w)
            start_y, end_y = self.get_crop_params(speed_y / (self.num_speeds // 2), h)
            intermediate_x = (torch.linspace(start_x, end_x, self.num_frames).long()).clamp_(0, w - self.crop_size)
            intermediate_y = (torch.linspace(start_y, end_y, self.num_frames).long()).clamp_(0, h - self.crop_size)

            frames_out = torch.empty(
                self.num_frames, self.crop_size, self.crop_size, c, device=video.device, dtype=video.dtype
            )

            for t in range(self.num_frames):
                frames_out[t] = selected_frame[
                                intermediate_y[t]:intermediate_y[t] + self.crop_size,
                                intermediate_x[t]:intermediate_x[t] + self.crop_size, :
                                ]

            # performs augmentation on the generated image sequence
            mock_input = {"video": frames_out}
            mock_output = self.transforms(mock_input)
            frames_out = mock_output["video"]

            # applies static mask
            if self.static_mask_enable:
                frames_out = self.static_mask(frames_out)
            out.append(frames_out)
        out = torch.stack(out)
        return out

    def label_generator(self):
        """ Generates the label for the MoSI.
        `separate` label is used for separate prediction on the two axes,
            i.e., two classification heads for each axis.
        'joint' label is used for joint prediction on the two axes.
        """
        if self.label_mode == 'separate':
            return self.generate_separate_labels()
        elif self.label_mode == 'joint':
            return self.generate_joint_labels()

    def generate_separate_labels(self):
        """ Generates labels for separate prediction.
        """
        label_x = []
        label_y = []
        for speed_idx, speed in enumerate(self.speed_set):
            speed_x, speed_y = speed
            speed_x_label = speed_x - self.speed_min - (speed_x > 0) * self.zero_out
            speed_y_label = speed_y - self.speed_min - (speed_y > 0) * self.zero_out
            label_x.append(speed_x_label)
            label_y.append(speed_y_label)
        return {
            "move_x": torch.tensor(label_x),
            "move_y": torch.tensor(label_y)
        }

    def generate_joint_labels(self):
        """ Generates labels for joint prediction.
        """
        return {
            "move_joint": torch.linspace(
                0, len(self.speed_set) - 1, len(self.speed_set), dtype=torch.int64
            )
        }

    def get_crop_params(self, speed_factor, total_length):
        """ Returns crop parameters.

        Args:
            speed_factor (float): frac{distance_to_go}{total_distance}
            total_length (int): length of the side
        """
        if speed_factor == 0:
            total_length >= self.crop_size, ValueError(
                "Total length ({}) should not be less than crop size ({}) for speed {}.".format(
                    total_length, self.crop_size, speed_factor
                ))
        else:
            assert total_length > self.crop_size, ValueError(
                "Total length ({}) should be larger than crop size ({}) for speed {}.".format(
                    total_length, self.crop_size, speed_factor
                ))
        assert abs(speed_factor) <= 1, ValueError(
            "Speed factor should be smaller than 1. But {} was given.".format(speed_factor))

        distance_factor = self.get_distance_factor(speed_factor) if self.mode == 'train' else 1
        distance = (total_length - self.crop_size) * speed_factor * distance_factor
        start_min = max(
            0, 0 - distance
        )  # if distance > 0, move right or down, start_x_min=0
        start_max = min(
            (total_length - self.crop_size),
            (total_length - self.crop_size) - distance
        )  # if distance > 0, move right or down, start_x_max = (w-crop_size)-distance
        start = random.randint(int(start_min), int(start_max)) if self.mode == 'train' else (
                                                                                                    total_length - self.crop_size - distance) // 2
        end = start + distance
        return start, end

    def get_distance_factor(self, speed_factor):
        # jitter on the distance
        if abs(speed_factor) < 1:
            distance_factor = random.uniform(self.distance_jitter[0], self.distance_jitter[1])
        else:
            distance_factor = random.uniform(self.distance_jitter[0], 1)
        return distance_factor

    def frame_size_standardize(self, frame):
        """ Standardize the frame size according to the settings in the cfg.
        Args:
            frame (torch.Tensor): A single frame with the shape of (C, 1, H, W) to be
                standardized.
        """
        h, w, _ = frame.shape
        if isinstance(self.standard_size, list):
            assert len(self.standard_size) == 3
            size_s, size_l, crop_size = self.standard_size
            reshape_size = random.randint(int(size_s), int(size_l))
        else:
            crop_size = self.standard_size
            reshape_size = self.standard_size

        # resize the short side to standard size
        dtype = frame.dtype
        frame = frame.permute(2, 0, 1).to(torch.float)  # C, H, W
        aspect_ratio = random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        if h <= w:
            new_h = reshape_size
            new_w = int(new_h / h * w)
            # resize
            frame = F.resize(frame.unsqueeze(0), (new_h, new_w), "bilinear").squeeze(0)
        else:
            new_w = reshape_size
            new_h = int(new_w / w * h)
            # resize
            frame = F.resize(frame.unsqueeze(0), (new_h, new_w), "bilinear").squeeze(0)

            # crop
        if aspect_ratio >= 1:
            crop_h = int(crop_size / aspect_ratio)
            crop_w = crop_size
        else:
            crop_h = crop_size
            crop_w = int(crop_size * aspect_ratio)
        start_h = random.randint(0, new_h - crop_h)
        start_w = random.randint(0, new_w - crop_w)
        return frame[:, start_h:start_h + crop_h, start_w:start_w + crop_w].to(dtype).permute(1, 2, 0)  # H, W, C

    def static_mask(self, frames):
        """ Apply static mask with random position and size to the generated pseudo motion sequence.
        Args:
            frames (torch.Tensor): (C, T, H, W).
        Returns:
            frames (torch.Tensor): Masked frames.
        """
        c, t, h, w = frames.shape
        rand_t = random.randint(0, t - 1)
        mask_size_ratio = random.uniform(self.mask_size_ratio[0], self.mask_size_ratio[1])
        mask_size_x, mask_size_y = [int(w * mask_size_ratio), int(h * mask_size_ratio)]
        start_x = random.randint(0, w - mask_size_x)
        start_y = random.randint(0, h - mask_size_y)
        frames_out = frames[:, rand_t].unsqueeze(1).expand(-1, t, -1, -1).clone()
        frames_out[:, :, start_y:start_y + mask_size_y, start_x:start_x + mask_size_x] = frames[
                                                                                         :, :,
                                                                                         start_y:start_y + mask_size_y,
                                                                                         start_x:start_x + mask_size_x
                                                                                         ]
        return frames_out

    def __call__(self, item):
        item[self.output_key] = self.sample_generator(item[self.input_key])
        item[self.out_label_key] = self.labels
        return item
