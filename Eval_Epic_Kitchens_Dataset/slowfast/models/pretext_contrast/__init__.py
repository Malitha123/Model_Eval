"""Defines PretextContrast network."""
from os import stat
import re
from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet
# from slowfast.models.pretext_contrast.network import R21D


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location="cpu")
    for name, params in pretrained_weights.items():
        if 'module' in name:
            name = name[name.find('module')+7:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


class PretextContrast(nn.Module):
    """PretextContrast Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PretextContrast, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)
        self.init_weights(cfg.MODEL.CKPT)

    def _construct_network(self, cfg):
        """
        PretextContrast

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=False)

        # add K = 2 heads (noun and verb prediction)
        # temporary hardcoding
        pool_size=[
            [4, 7, 7]
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
    
    @staticmethod
    def find_key(pattern, query, values):
        """Returns all strings in queries (list of strings) which have the pattern."""
        return [x for x in values if re.search(pattern, x)]
    
    @staticmethod
    def modify_key(key):
        start_char = key.split(".")[0]
        if start_char.isdigit():

            if int(start_char) == 0:
                key = key.replace("0.", "stem.", 1)

            if int(start_char) > 0:
                key = key.replace(f"{start_char}.", f"layer{start_char}.", 1)

        return key
    
    def init_weights(self, ckpt_path):
        assert isfile(ckpt_path), f"Checkpoint does not exist at {ckpt_path}."
        # pretrained_weights = load_pretrained_weights(ckpt_path)
        pretrained_weights = torch.load(ckpt_path, map_location="cpu")
        pretrained_weights = {k.replace("module.base_network.", ""):v for k, v in pretrained_weights.items()}
        pretrained_weights = {self.modify_key(k):v for k, v in pretrained_weights.items()}

        pt_keys = list(pretrained_weights.keys())
        en_keys = list(self.encoder.state_dict().keys())

        intersection = set(pretrained_weights).intersection(set(en_keys))
        print(f"\n::::: Found intersection of {len(intersection)} keys between checkpoint and encoder. \n")
        
        msg = self.encoder.load_state_dict(pretrained_weights, strict=False)
        print(msg)
        print(f"\n::::: Loaded pretrained weights from {ckpt_path} \n")


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    args.cfg_file = "../../../configs/EPIC-KITCHENS/PRETEXT_CONTRAST/32x112x112_PC_R18_K400_LR0.0025.yaml"
    cfg = load_config(args)

    # load model
    model = PretextContrast(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

