import os
import pandas as pd
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets.epickitchens_record import EpicKitchensVideoRecord

from slowfast.datasets import transform as transform
from slowfast.datasets import utils as utils
from slowfast.datasets.frame_loader import pack_frames_to_video_clip

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = 60
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        
        if self.cfg.DATA.USE_UNIQUE_FRAMES:
            num_unique_frames = self.cfg.DATA.NUM_UNIQUE_FRAMES      
            logger.info(f"::: Using unique frames: {num_unique_frames}")

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._video_records.append(EpicKitchensVideoRecord(tup))
                    self._spatial_temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )
        
        # filter N number of samples
        num_samples = self.cfg.DATA.get("NUM_SAMPLES", None)
        if num_samples is not None and self.mode == "train":
            video_names = [v.metadata["narration_id"] for v in self._video_records]
            video_labels = [v.label["verb"] for v in self._video_records]

            subset_indices = utils.get_subset_data(
                video_names, video_labels, num_samples, seed=self.cfg.RNG_SEED,
            )
            self._video_records = [
                self._video_records[i] for i in subset_indices
            ]
            self._spatial_temporal_idx = [
                self._spatial_temporal_idx[i] for i in subset_indices
            ]
            logger.info("Filtered {} samples".format(len(self._video_records)))

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)
        if self.cfg.DATA.USE_UNIQUE_FRAMES:
            import numpy as np
            num_unique_frames = self.cfg.DATA.NUM_UNIQUE_FRAMES
            
            frame_idx = np.random.choice(frames.shape[0], num_unique_frames, replace=False)
            frame_idx = np.repeat(frame_idx, frames.shape[0] // num_unique_frames)
            frames = frames[frame_idx]
            
            logger.info(f"::: Using unique frames: {num_unique_frames}, shape: {frames.shape}")
        
        # Perform color normalization.
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        label = self._video_records[index].label
        frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config
    import os
    from os.path import join, dirname, abspath

    repo_path = dirname(dirname(dirname(abspath(__file__))))
    cfg_path = join(repo_path, "configs/EPIC-KITCHENS/R2PLUS1D/8x112x112_R18_K400_LR0.0025_uniq_frames_1.yaml")
    
    hostname = os.uname()[1]
    if hostname == "diva":
        dataset_dir = "/ssd/pbagad/datasets/EPIC-KITCHENS-100/EPIC-KITCHENS/"
        annotations_dir = "/ssd/pbagad/datasets/EPIC-KITCHENS-100/annotations/"
    elif hostname == "fs4":
        dataset_dir = "/var/scratch/pbagad/EPIC-KITCHENS/"
        annotations_dir = "/var/scratch/pbagad/datasets/EPIC-KITCHENS-100/annotations/"
    else:
        raise ValueError("Invalid hostname: {}".format(hostname))

    args = parse_args()
    args.cfg_file = cfg_path
    cfg = load_config(args)
    
    cfg.EPICKITCHENS.VISUAL_DATA_DIR = dataset_dir
    cfg.EPICKITCHENS.ANNOTATIONS_DIR = annotations_dir
    cfg.DATA.NUM_SAMPLES = 1000
    
    dataset = Epickitchens(cfg, "train")
    
    frames, label, index, metadata = dataset[0]
    assert len(frames) == 1
    assert frames[0].shape == torch.Size([3, 8, 112, 112])
    assert label == {'verb': 0, 'noun': 58}
    assert metadata == {'narration_id': 'P22_15_137'}

