#!/bin/bash
#SBATCH --job-name=LastfineTune
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:16
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu


time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints_linear/gdt/ --seed 100 > ./jobs/outs/gdt_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints_linear/ctp/ --seed 100 > ./jobs/outs/ctp_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_tclr_linear_eval.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints_linear/tclr/ --seed 100 > ./jobs/outs/tclr_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints_linear/rspnet/ --seed 100 > ./jobs/outs/rspnet_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints_linear/video_moco/ --seed 100 > ./jobs/outs/video_moco_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_linear_eval.yaml  --pretext-model-name  selavi  --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints_linear/selavi/ --seed 100 > ./jobs/outs/selavi_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python linear_eval.py   configs/benchmark/something/eval/112x112x32_avid_linear_eval.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints_linear/avid_cma/ --seed 100 > ./jobs/outs/avid_cma_linear_finetune_out_$time.out
