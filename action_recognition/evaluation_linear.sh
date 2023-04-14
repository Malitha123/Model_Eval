#!/bin/bash
#SBATCH --job-name=EvalLinear
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:16
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml       --pretext-model-name  pretext_contrast    --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt          --finetune-ckpt-path ./training_checkpoints/ssv2/linear/pretext_contrast/  > ./jobs/linear_fine_tune_results/pretext_contrast_linear_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml      --pretext-model-name  moco          --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar                     --finetune-ckpt-path ./training_checkpoints/ssv2/linear/moco/ > ./jobs/linear_fine_tune_results/moco_linear_finetune_out_$time.out


time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/ctp/ > ./jobs/linear_fine_tune_results/ctp_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/gdt/ > ./jobs/linear_fine_tune_results/gdt_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_tclr_linear_evaluation_test.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/tclr/ > ./jobs/linear_fine_tune_results/tclr_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/rspnet/  > ./jobs/linear_fine_tune_results/rspnet_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/video_moco/  > ./jobs/linear_fine_tune_results/video_moco_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py   configs/benchmark/something/eval/112x112x32_linear_evaluation_test.yaml  --pretext-model-name  selavi --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./training_checkpoints/ssv2/linear/selavi/  > ./jobs/linear_fine_tune_results/selavi_linear_finetune_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test.py  configs/benchmark/something/eval/112x112x32_avid_linear_evaluation_test.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/linear/avid_cma/ > ./jobs/linear_fine_tune_results/avid_cma_linear_finetune_eval_out_$time.out
