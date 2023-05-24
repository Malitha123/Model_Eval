"""Defines GDT model."""
from os.path import join, exists
import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from slowfast.models.r2plus1d import video_resnet
# from slowfast.models.gdt.model import VideoBaseNetwork, Identity, 
from slowfast.models.gdt.model import random_weight_init, load_model_parameters
from slowfast.models import head_helper


class GDTBase(nn.Module):
    """GDT Model for EPIC-KITCHENS finetuning."""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(GDTBase, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        GDT

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        pretrained = False
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=pretrained)
        random_weight_init(self.encoder)

        # load encoder weights from pretrained ckpt
        ckpt_path = cfg.MODEL.CKPT
        assert ckpt_path is not None, "Checkpoint path is not present in config: cfg.MODEL.CKPT"
        self.init_weights_from_ckpt(ckpt_path)

        # NOTE: Not using VideoBaseNetwork since it squeezes the first dimension of input,
        # it has avgpool and FC which we do not need

        # self.encoder = VideoBaseNetwork(vid_base_arch=cfg.MODEL.ARCH, pretrained=pretrained, pre_pool=True)
        # self.encoder.avgpool = Identity()
        # self.encoder.fc = Identity()

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
    
    def init_weights_from_ckpt(self, ckpt_path: str):
        """Loads checkpoint and sets model encoder weights from it."""
        assert exists(ckpt_path), f"Checkpoint does not exist for GDTBase at {ckpt_path}"

        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        key = "state_dict" if "state_dict" in ckpt_dict else "model"
        ckpt_weights = ckpt_dict[key]
        epoch = ckpt_dict["epoch"]
        print(f":::: Epoch checkpoint: {epoch}")

        # change keys of ckpt_weights that are relevant
        relevant_ckpt_weights = {k.replace("video_network.base.", ""):v for k, v in ckpt_weights.items() if k.startswith("video_network.base.")}
        relevant_ckpt_keys = list(relevant_ckpt_weights.keys())
        relevant_model_keys = list(self.encoder.state_dict().keys())
        len_intersection = len(set(relevant_model_keys).intersection(set(relevant_ckpt_keys)))
        assert len_intersection
        print(f":::: Loading checkpoint: {len(relevant_model_keys)} model keys and {len(relevant_ckpt_keys)} checkpoint keys.")

        load_model_parameters(self.encoder, relevant_ckpt_weights)


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    args.cfg_file = "../../../configs/EPIC-KITCHENS/GDT/base.yaml"
    cfg = load_config(args)

    # load model
    model = GDTBase(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)

    # check encoder shape
    assert model.encoder(x).shape == torch.Size([1, 512, 4, 7, 7])

    # check final output shape
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

    # check if loaded model weights are correct
    ckpt_path = "/var/scratch/pbagad/AVID-CMA_Refactored/checkpoints/GDT/Kinetics/pretext_checkpoint/gdt_K400.pth"
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    key = "state_dict" if "state_dict" in ckpt_dict else "model"
    ckpt_weights = ckpt_dict[key]

    layer_to_check = "layer4.0.conv2.0.0.weight"
    model_layer_weights = model.encoder.state_dict()[layer_to_check]
    ckpt_layer_weights = ckpt_weights[f"video_network.base.{layer_to_check}"]
    assert (ckpt_layer_weights == model_layer_weights).all()

