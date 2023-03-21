#!/bin/bash
#SBATCH --job-name=EvalFFT
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu

#SSV2

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ > ./jobs/full_fine_tune_evaluation_results/gdt_full_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ > ./jobs/full_fine_tune_evaluation_results/ctp_full_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_tclr_fine_tune_eval.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ > ./jobs/full_fine_tune_evaluation_results/tclr_full_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ > ./jobs/full_fine_tune_evaluation_results/rspnet_full_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  selavi --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/  > ./jobs/full_fine_tune_evaluation_results/selavi_full_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_avid_fine_tune_eval.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ > ./jobs/full_fine_tune_evaluation_results/avid_cma_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/  > ./jobs/full_fine_tune_evaluation_results/video_moco_full_finetune_eval_out_$time.out


## NTU

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ > ./jobs/full_fine_tune_evaluation_results/gdt_full_finetune_eval_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./checkpoints/ctp/ > ./jobs/full_fine_tune_evaluation_results/ctp_full_finetune_eval_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_tclr_fine_tune_eval.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ > ./jobs/full_fine_tune_evaluation_results/tclr_full_finetune_eval_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ > ./jobs/full_fine_tune_evaluation_results/rspnet_full_finetune_eval_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml  --pretext-model-name  selavi --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./checkpoints/selavi/  > ./jobs/full_fine_tune_evaluation_results/selavi_full_finetune_eval_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_avid_fine_tune_eval.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./checkpoints/avid_cma/ > ./jobs/full_fine_tune_evaluation_results/avid_cma_full_finetune_out_$time.out

# time=`date +%m-%d_%H-%M-%S`
# /nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_fine_tune_eval.yaml --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/  > ./jobs/full_fine_tune_evaluation_results/video_moco_full_finetune_eval_out_$time.out



 