"""Code for TCLR method."""


from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet


class TCLR(nn.Module):
    """TCLR"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(TCLR, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        TCLR network.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        pretrained = cfg.MODEL.PRETRAINED
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=pretrained)
        self.encoder.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                    stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
        self.encoder.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                            kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)


        self.init_weights_from_checkpoint(cfg.MODEL.CKPT)

        # add K = 2 heads (noun and verb prediction)
        # temporary hardcoding
        # this needs (8, 7, 7) pooling instead of (4, 7, 7) due to the change in the last conv layer
        pool_size=[
            [8, 7, 7]
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
    
    def init_weights_from_checkpoint(self, checkpoint_file):
        """Inits weights from given checkpoint."""
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        csd = ckpt["state_dict"]

        csd_items = csd.items()
        csd_subset = dict()
        for layer_name, weights in csd_items:
            if 'module.1.' in layer_name:
                continue              
            elif '1.' == layer_name[:2]:
                continue
            if 'module.0.' in layer_name:
                layer_name = layer_name.replace('module.0.','')
            if 'module.' in layer_name:
                layer_name = layer_name.replace('module.','')
            elif '0.' == layer_name[:2]:
                layer_name = layer_name[2:]
            if 'fc' in layer_name:
                continue
            csd_subset[layer_name] = weights

        msg = self.encoder.load_state_dict(csd_subset, strict=False)

        print(":::::::::::::::::::::: Loaded pretrained checkpoint from ::::::::::::::::::::::")
        print(f"Path:\t {checkpoint_file}")
        print(f"Message:\t {msg}")

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
    args.cfg_file = "../../../configs/EPIC-KITCHENS/TCLR/tclr_32x112x112_R18_K400_LR0.0025.yaml"
    cfg = load_config(args)

    # load model
    model = TCLR(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])

    assert y[0].shape == (1, 97)
    assert y[1].shape == (1, 300)

