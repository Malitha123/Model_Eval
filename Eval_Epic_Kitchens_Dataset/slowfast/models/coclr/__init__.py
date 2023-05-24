"""Defines class for CoCLR model (MoCo)."""
from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet


class CoCLR(nn.Module):
    """CoCLR"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(CoCLR, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        R(2 + 1)D backbone.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=False)
        self.init_weights(ckpt_path=cfg.MODEL.CKPT)

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

    def init_weights(self, ckpt_path: str):
        if isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            # import ipdb; ipdb.set_trace()

            new_dict = {}
            for k,v in state_dict.items():
                if k.startswith("encoder_k"):
                    continue
                k = k.replace('encoder_q.0.', '')
                new_dict[k] = v
            state_dict = new_dict

            CSD = set(state_dict.keys())
            ESD = set(self.encoder.state_dict().keys())

            #try: model.load_state_dict(state_dict)
            #except: neq_load_customized(model_without_dp, state_dict, verbose=True)
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(msg)
            print("")
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(ckpt_path))
            raise NotImplementedError


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    args.cfg_file = "../../../configs/EPIC-KITCHENS/COCLR/das5_32x112x112_R18_K400_LR0.0025_linear.yaml"
    cfg = load_config(args)

    # load model
    model = CoCLR(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

