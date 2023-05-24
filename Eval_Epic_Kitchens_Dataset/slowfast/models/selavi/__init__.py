"""Implemented R(2 + 1)D network."""

from os.path import isfile, exists

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet

from slowfast.models.selavi.model import load_model
from slowfast.models.selavi.model import load_model_parameters


class SELAVI(nn.Module):
    """Paper: https://eml-workshop.github.io/Papers/B4.pdf"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SELAVI, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        SELAVI network.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        model = load_model(
            vid_base_arch=cfg.MODEL.ARCH,
            aud_base_arch="resnet9",
            pretrained=False,
            num_classes=400,
            norm_feat=False,
            use_mlp=True,
            headcount=10,
        )
        self.init_weights_from_ckpt(model, cfg.MODEL.CKPT)
        model.video_network.base.avgpool = nn.Identity()
        self.encoder = model.video_network.base

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
    
    def init_weights_from_ckpt(self, model, ckpt_path):
        """Loads encoder weights."""
        if exists(ckpt_path):
            print("Loading model weights")
            ckpt_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
            try:
                model_weights = ckpt_dict["state_dict"]
            except:
                model_weights = ckpt_dict["model"]
            epoch = ckpt_dict["epoch"]
            print(f"Epoch checkpoint: {epoch}")
            load_model_parameters(model, model_weights)
            print(f"Loading model done")
        else:
            print(f"Training from scratch")

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
    args.cfg_file = "../../../configs/EPIC-KITCHENS/SELAVI/32x112x112_R18_K400_LR0.0025.yaml"
    cfg = load_config(args)

    # load model
    model = SELAVI(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

