#!/bin/bash
# EPIC-KITCHENS

# 1 - pretext_contrast
cfg_path=configs/EPIC-KITCHENS/PRETEXT_CONTRAST/32x112x112_PC_R18_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4

# 2 - moco
cfg_path=configs/EPIC-KITCHENS/MOCO/diva_32x112x112_R18_K400_LR0.0025.yaml 
bash das5_train.sh -c $cfg_path -n 4

# 3 - gdt
cfg_path=configs/EPIC-KITCHENS/GDT/32x112x112_GDT_R2+1D_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4

# 4 - ctp
cfg_path=configs/EPIC-KITCHENS/CTP/32x112x112_R2+1D-18_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4

# 5 - tcrl
cfg_path=configs/EPIC-KITCHENS/TCRL/tclr_32x112x112_R18_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4

# 6 - rspnet
cfg_path=configs/EPIC-KITCHENS/RSPNET/diva_32x112x112_R18_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4

# 7 - video_moco
cfg_path=configs/EPIC-KITCHENS/VIDEOMOCO/32x112x112_VMOCO_R18_K400_LR0.0025.yaml 
bash das5_train.sh -c $cfg_path -n 4

# 8 - selavi
cfg_path=configs/EPIC-KITCHENS/SELAVI/selavi_32x112x112_R18_K400_LR0.0025.yaml 
bash das5_train.sh -c $cfg_path -n 4

# 9 - avid_cma
cfg_path=configs/EPIC-KITCHENS/AVID_CMA/32x112x112_R18_K400_LR0.0025.yaml
bash das5_train.sh -c $cfg_path -n 4
