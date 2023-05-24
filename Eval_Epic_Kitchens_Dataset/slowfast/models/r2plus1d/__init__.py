"""Implemented R(2 + 1)D network."""

from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet


class R2Plus1D(nn.Module):
    """R(2+1)D Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(R2Plus1D, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        R(2 + 1)D

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        pretrained = cfg.MODEL.PRETRAINED
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=pretrained)

        # add K = 2 heads (noun and verb prediction)
        # temporary hardcoding
        pool_size=[
            # [4, 7, 7]
            [cfg.DATA.NUM_FRAMES // 8, 7, 7]
        ]

        # temporary hardcoding
        self.head = head_helper.ResNetBasicHead(
            dim_in=[512],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=pool_size,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    def forward(self, x):

        for pathway in range(self.num_pathways):
            x[pathway] = self.encoder(x[pathway])

        x = self.head(x)
        return x

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'bn_parameters':
            print("Freezing all BN layers\' parameters.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == 'bn_statistics':
            print("Freezing all BN layers\' statistics.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown running statistics update in frozen mode
                    m.eval()


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    args.cfg_file = "../../../configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_scratch_LR0.0025.yaml"
    cfg = load_config(args)

    # load model
    model = R2Plus1D(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

