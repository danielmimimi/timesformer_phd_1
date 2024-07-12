# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import glob
import random
from itertools import chain as chain
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Hslu(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 20
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Loading label names.
        with PathManager.open(
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "label_files.json",
            ),
            "r",
        ) as f:
            label_dict = json.load(f)

        # Loading labels.
        if self.mode == "val":
            label_file = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "annotations_validation_{}.json".format(self.cfg.DATA.SAMPLING_METHOD),
            )
        else:
            label_file = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "annotations_{}_{}.json".format(
                    self.mode,self.cfg.DATA.SAMPLING_METHOD
                ),
            )
                        
        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        self._start = []
        self._stop = []
        for video in label_json['Data']:
            video_name = video["id"]
            label = video["label"]
            self._video_names.append(video_name)
            self._labels.append(label)
            self._start.append(video["start"])
            self._stop.append(video["stop"])

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format("train" if self.mode == "train" else "val"),
        )


        self._path_to_videos = []

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_names]
            )
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val","test"]: #or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            #self._spatial_temporal_idx[index]
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )

            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        label = self._labels[index]
        start_index = self._start[index]
        stop_index = self._stop[index]
        path_to_video_frames = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,self._video_names[index])
        
        entire_video_frames = glob.glob(os.path.join(path_to_video_frames, 'image_*.png'))
        if len(entire_video_frames) == 0:
             entire_video_frames = glob.glob(os.path.join(path_to_video_frames, 'image_*.jpg'))
        num_frames = self.cfg.DATA.NUM_FRAMES
        def extract_frame_number(image_path):
            filename = image_path.split('/')[-1]  # Get the filename
            frame_number = int(filename.split('_')[1].split('.')[0])  # Extract frame number
            return frame_number
        frame_to_path = {extract_frame_number(path): path for path in entire_video_frames}
        selected_frames = [frame_to_path[i] for i in range(start_index, stop_index + 1) if i in frame_to_path]

        video_length = len(selected_frames)

        seg_size = float(video_length - 1) / num_frames
        
        seq = []

        if self.cfg.DATA.SAMPLING_METHOD == "uniform":
            if label == 4:
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    # if self.mode == "train":
                    seq.append(random.randint(start, end))
        elif self.cfg.DATA.SAMPLING_METHOD == "one_step": # around 18 frames for completion - around step 2/3 frame dif
            
            if label == 4:     
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                try:
                    random_selector = random.randint(0, 1)
                    if video_length < 24:
                        random_selector = 0
                    if random_selector == 0:
                        start_point = random.randint(0, video_length - num_frames*2)
                        seq = [start_point + i* 2 for i in range(num_frames)]
                    else:
                        start_point = random.randint(0, video_length - num_frames*3)
                        seq = [start_point + i* 3 for i in range(num_frames)]
                except:
                    pass
        elif self.cfg.DATA.SAMPLING_METHOD == "half_step": # around 8 frames for completion - around step 1/2 frame dif        
            if label == 4:     
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                try:
                    random_selector = random.randint(0, 1)
                    if video_length < 16:
                        random_selector = 0
                    if random_selector == 0:
                        start_point = random.randint(0, video_length - num_frames*1)
                        seq = [start_point + i* 1 for i in range(num_frames)]
                    else:
                        start_point = random.randint(0, video_length - num_frames*2)
                        seq = [start_point + i*2 for i in range(num_frames)]
                except:
                    pass
        else:
            raise Exception("SAMPLING METHOD NOT EXISTING")
        self.prev_seq = seq
        # random_selector = random.randint(0, 1)
        # if random_selector == 0:
        #     seg_size = float(video_length - 1) / num_frames
        #     seq = []
        #     for i in range(num_frames):
        #         start = int(np.round(seg_size * i))
        #         end = int(np.round(seg_size * (i + 1)))
        #         # if self.mode == "train":
        #         seq.append(random.randint(start, end))
        #         # else:
        #         #     seq.append((start + end) // 2)
        # else:
        #     try:
        #         start_point = random.randint(0, video_length - num_frames)
        #         seq = [start_point + i for i in range(num_frames)]
        #     except:
        #         pass

        # seg_size = float(video_length - 1) / num_frames
        # seq = []
        # for i in range(num_frames):
        #     start = int(np.round(seg_size * i))
        #     end = int(np.round(seg_size * (i + 1)))
        #     if self.mode == "train":
        #         seq.append(random.randint(start, end))
        #     else:
        #         seq.append((start + end) // 2)
        # ADD A SORT FOR DIRECTION
        frames = torch.as_tensor(
            utils.retry_load_images(
                [selected_frames[frame] for frame in seq],
                self._num_retries,
            )
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        first_frame = frames[0]
        frames = frames.permute(3, 0, 1, 2)

        frames = utils.simple_scale(frames,crop_size)
        # frames = utils.spatial_sampling(
        #     frames,
        #     spatial_idx=spatial_sample_index,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        #     crop_size=crop_size,
        #     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
        #     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        # )
        #if not self.cfg.RESFORMER.ACTIVE:
        if not self.cfg.MODEL.ARCH in ['vit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            # Perform temporal sampling from the fast pathway.
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )
        if self.mode == "test":
            return frames, label, index,{"frame_paths":[selected_frames[frame] for frame in seq]}
        else:
            return frames, label, index,{}
        

    def __get_frame_info__(self,index):
        label = self._labels[index]
        start_index = self._start[index]
        stop_index = self._stop[index]
        path_to_video_frames = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,self._video_names[index])
        return start_index,stop_index

    def __getitem_rolling__(self, index,external_start_index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val","test"]: #or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            #self._spatial_temporal_idx[index]
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )

            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        label = self._labels[index]
        start_index = self._start[index]
        stop_index = self._stop[index]
        path_to_video_frames = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,self._video_names[index])
        
        entire_video_frames = glob.glob(os.path.join(path_to_video_frames, 'image_*.png'))
        if len(entire_video_frames) == 0:
            entire_video_frames = glob.glob(os.path.join(path_to_video_frames, 'image_*.jpg'))
        num_frames = self.cfg.DATA.NUM_FRAMES
        def extract_frame_number(image_path):
            filename = image_path.split('/')[-1]  # Get the filename
            frame_number = int(filename.split('_')[1].split('.')[0])  # Extract frame number
            return frame_number
        frame_to_path = {extract_frame_number(path): path for path in entire_video_frames}
        # selected_frames = [frame_to_path[i] for i in range(start_index, stop_index + 1) if i in frame_to_path]

        video_length = start_index- stop_index

        seg_size = float(video_length - 1) / num_frames
        
        seq = []

        if self.cfg.DATA.SAMPLING_METHOD == "uniform":
            if label == 4:
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    # if self.mode == "train":
                    seq.append(random.randint(start, end))
        elif self.cfg.DATA.SAMPLING_METHOD == "one_step": # around 18 frames for completion - around step 2/3 frame dif
            
            if label == 4:     
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                try:
                    random_selector = random.randint(0, 1)
                    if video_length < 24:
                        random_selector = 0
                    if random_selector == 0:
                        start_point = random.randint(0, video_length - num_frames*2)
                        seq = [start_point + i* 2 for i in range(num_frames)]
                    else:
                        start_point = random.randint(0, video_length - num_frames*3)
                        seq = [start_point + i* 3 for i in range(num_frames)]
                except:
                    pass
        elif self.cfg.DATA.SAMPLING_METHOD == "half_step": # around 8 frames for completion - around step 1/2 frame dif        
            if label == 4:     
                start_point = random.randint(0, video_length - num_frames)
                seq = [start_point + i for i in range(num_frames)]
            else:
                try:
                    random_selector = random.randint(0, 1)
                    if video_length < 16:
                        random_selector = 0
                    if random_selector == 0:
                        start_point = random.randint(0, video_length - num_frames*1)
                        seq = [start_point + i* 1 for i in range(num_frames)]
                    else:
                        start_point = random.randint(0, video_length - num_frames*2)
                        seq = [start_point + i*2 for i in range(num_frames)]
                except:
                    pass
        elif self.cfg.DATA.SAMPLING_METHOD == "rolling": # around 8 frames for completion - around step 1/2 frame dif        
            try:
                seq = [external_start_index + i* 1 for i in range(num_frames)]
            except:
                pass
        else:
            raise Exception("SAMPLING METHOD NOT EXISTING")
        self.prev_seq = seq
        # random_selector = random.randint(0, 1)
        # if random_selector == 0:
        #     seg_size = float(video_length - 1) / num_frames
        #     seq = []
        #     for i in range(num_frames):
        #         start = int(np.round(seg_size * i))
        #         end = int(np.round(seg_size * (i + 1)))
        #         # if self.mode == "train":
        #         seq.append(random.randint(start, end))
        #         # else:
        #         #     seq.append((start + end) // 2)
        # else:
        #     try:
        #         start_point = random.randint(0, video_length - num_frames)
        #         seq = [start_point + i for i in range(num_frames)]
        #     except:
        #         pass

        # seg_size = float(video_length - 1) / num_frames
        # seq = []
        # for i in range(num_frames):
        #     start = int(np.round(seg_size * i))
        #     end = int(np.round(seg_size * (i + 1)))
        #     if self.mode == "train":
        #         seq.append(random.randint(start, end))
        #     else:
        #         seq.append((start + end) // 2)
        # ADD A SORT FOR DIRECTION
        frames = torch.as_tensor(
            utils.retry_load_images(
                [frame_to_path[frame] for frame in seq],
                self._num_retries,
            )
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        first_frame = frames[0]
        frames = frames.permute(3, 0, 1, 2)

        frames = utils.simple_scale(frames,crop_size)
        # frames = utils.spatial_sampling(
        #     frames,
        #     spatial_idx=spatial_sample_index,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        #     crop_size=crop_size,
        #     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
        #     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        # )
        #if not self.cfg.RESFORMER.ACTIVE:
        if not self.cfg.MODEL.ARCH in ['vit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            # Perform temporal sampling from the fast pathway.
            frames = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                    ).long(),
            )
        if self.mode == "test":
            return frames, label, index,{"frame_paths":[frame_to_path[frame] for frame in seq]}
        else:
            return frames, label, index,{}
        
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_names)
