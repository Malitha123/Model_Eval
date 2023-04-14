#!/bin/bash
#SBATCH --job-name=LastfineTune
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu

# # SSV2

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py  configs/benchmark/something/eval/112x112x32_linear_eval.yaml       --pretext-model-name  pretext_contrast    --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt          --finetune-ckpt-path ./training_checkpoints/ssv2/linear/pretext_contrast/ --seed 100 > ./jobs/outs/pretext_contrast_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py  configs/benchmark/something/eval/112x112x32_linear_eval.yaml       --pretext-model-name  moco          --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar                     --finetune-ckpt-path ./training_checkpoints/ssv2/linear/moco/ --seed 100 > ./jobs/outs/moco_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/gdt/ --seed 100 > ./jobs/outs/gdt_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/ctp/ --seed 100 > ./jobs/outs/ctp_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_tclr_linear_eval.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/tclr/ --seed 100 > ./jobs/outs/tclr_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/rspnet/ --seed 100 > ./jobs/outs/rspnet_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/video_moco/ --seed 100 > ./jobs/outs/video_moco_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/selavi/ --seed 100 > ./jobs/outs/selavi_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_avid_linear_eval.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/avid_cma/ --seed 100 > ./jobs/outs/avid_cma_linear_finetune_out_$time.out



# # NTU60
# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./training_checkpoints/ntu/linear/gdt/ --seed 100 > ./jobs/outs/gdt_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./training_checkpoints/ntu/linear/ctp/ --seed 100 > ./jobs/outs/ctp_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./training_checkpoints/ntu/linear/tclr/ --seed 100 > ./jobs/outs/tclr_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/linear/rspnet/ --seed 100 > ./jobs/outs/rspnet_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/linear/video_moco/ --seed 100 > ./jobs/outs/video_moco_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./training_checkpoints/ntu/linear/selavi/ --seed 100 > ./jobs/outs/selavi_linear_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/ntu60/train/linear.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./training_checkpoints/ntu/linear/avid_cma/ --seed 100 > ./jobs/outs/avid_cma_linear_finetune_out_$time.out



