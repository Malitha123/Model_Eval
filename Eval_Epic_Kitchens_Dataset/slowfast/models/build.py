#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry


from slowfast.models.r2plus1d import R2Plus1D
from slowfast.models.gdt import GDTBase
from slowfast.models.ctp import CTP
from slowfast.models.videomoco import VideoMoCo
from slowfast.models.pretext_contrast import PretextContrast
from slowfast.models.tclr import TCLR
from slowfast.models.selavi import SELAVI
from slowfast.models.avid_cma import AVID_CMA
from slowfast.models.rspnet import RSPNet
from slowfast.models.coclr import CoCLR
from slowfast.models.moco import MOCO


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

MODEL_REGISTRY._do_register("CTP", CTP)
MODEL_REGISTRY._do_register("R2Plus1D", R2Plus1D)
MODEL_REGISTRY._do_register("GDTBase", GDTBase)
MODEL_REGISTRY._do_register("VideoMoCo", VideoMoCo)
MODEL_REGISTRY._do_register("PretextContrast", PretextContrast)
MODEL_REGISTRY._do_register("TCLR", TCLR)
MODEL_REGISTRY._do_register("SELAVI", SELAVI)
MODEL_REGISTRY._do_register("AVID_CMA", AVID_CMA)
MODEL_REGISTRY._do_register("RSPNet", RSPNet)
MODEL_REGISTRY._do_register("CoCLR", CoCLR)
MODEL_REGISTRY._do_register("MOCO", MOCO)


def build_model(cfg):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
    """
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)

    if cfg.MODEL.FREEZE_BACKBONE:
        print("Freezing backbone")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    return model
