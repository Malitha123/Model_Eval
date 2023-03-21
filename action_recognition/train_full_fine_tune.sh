#!/bin/bash
#SBATCH --job-name=FullFineTune
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu


# # SSV2
time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100 > ./jobs/outs/gdt_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ --seed 100 > ./jobs/outs/ctp_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32_tclr.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ --seed 100 > ./jobs/outs/tclr_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ --seed 100 > ./jobs/outs/rspnet_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ --seed 100 > ./jobs/outs/video_moco_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32.yaml --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/ --seed 100 > ./jobs/outs/selavi_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/something/train/112x112x32_avid.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ --seed 100 > ./jobs/outs/avid_cma_full_finetune_out_$time.out


## NTU

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./training_checkpoints/ntu/gdt/ --seed 100 --resume > ./jobs/outs/gdt_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./training_checkpoints/ntu/ctp/ --seed 100 --resume > ./jobs/outs/ctp_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./training_checkpoints/ntu/tclr/ --seed 100 > ./jobs/outs/tclr_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/rspnet/ --seed 100 > ./jobs/outs/rspnet_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/video_moco/ --seed 100 > ./jobs/outs/video_moco_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./training_checkpoints/ntu/selavi/ --seed 100 > ./jobs/outs/selavi_full_finetune_out_NTU_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configs/benchmark/ntu60/train/FFT.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/avid_cma/ --seed 100 > ./jobs/outs/avid_cma_full_finetune_out_NTU_$time.out

