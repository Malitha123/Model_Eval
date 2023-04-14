#!/bin/bash
#SBATCH --job-name=DirectEval
#SBATCH --output=/nfs/users/ext_malitha.gunawardhana/SEVERE-BENCHMARK/action_recognition/jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu

# # SSV2
time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_direct_eval.yaml       --pretext-model-name  pretext_contrast    --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt          --finetune-ckpt-path ./training_checkpoints/ssv2/direct/pretext_contrast/  > ./jobs/full_fine_tune_evaluation_results/pretext_contrast_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_direct_eval.yaml      --pretext-model-name  moco          --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar                     --finetune-ckpt-path ./training_checkpoints/ssv2/direct/moco/ > ./jobs/full_fine_tune_evaluation_results/moco_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_direct_eval.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/direct/gdt/ > ./jobs/direct_evaluation_results/gdt_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_direct_eval.yaml  --pretext-model-name  ctp --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth --finetune-ckpt-path ./training_checkpoints/ssv2/direct/ctp/ > ./jobs/direct_evaluation_results/ctp_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_tclr_direct_eval.yaml  --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./training_checkpoints/ssv2/direct/tclr/ > ./jobs/direct_evaluation_results/tclr_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_direct_eval.yaml  --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/direct/rspnet/ > ./jobs/direct_evaluation_results/rspnet_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py   configs/benchmark/something/eval/112x112x32_direct_eval.yaml  --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/direct/video_moco/  > ./jobs/direct_evaluation_results/video_moco_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py   configs/benchmark/something/eval/112x112x32_direct_eval.yaml  --pretext-model-name  selavi --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth --finetune-ckpt-path ./training_checkpoints/ssv2/direct/selavi/  > ./jobs/direct_evaluation_results/selavi_direct_eval_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python test_direct_eval.py  configs/benchmark/something/eval/112x112x32_avid_direct_eval.yaml  --pretext-model-name  avid_cma --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar --finetune-ckpt-path ./training_checkpoints/ssv2/direct/avid_cma/ > ./jobs/direct_evaluation_results/avid_cma_direct_eval_out_$time.out  
