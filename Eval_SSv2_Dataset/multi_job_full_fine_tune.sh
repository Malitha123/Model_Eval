#!/bin/bash
#SBATCH --job-name=allLayers
#SBATCH --output=./jobs/slurms/slurmjob.%J.out
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH --partition=multigpu


## SSV2

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  pretext_contrast    --pretext-model-path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt          --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/pretext_contrast/ --resume --seed 100 > ./jobs/outs/pretext_contrast_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  moco          --pretext-model-path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar                     --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/moco/ --resume --seed 100 > ./jobs/outs/moco_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  gdt           --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth                                --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/gdt/ --resume --seed 100 > ./jobs/outs/gdt_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  ctp           --pretext-model-path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth      --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/ctp/ --resume --seed 100 > ./jobs/outs/ctp_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32_tclr.yaml  --pretext-model-name  tclr          --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth                            --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/tclr/ --resume  --seed 100 > ./jobs/outs/tclr_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  rspnet        --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar    --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/rspnet/  --resume --seed 100 > ./jobs/outs/rspnet_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  video_moco    --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar     --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/video_moco/ --resume  --seed 100 > ./jobs/outs/video_moco_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  selavi        --pretext-model-path ../checkpoints_pretraining/selavi/selavi_kinetics.pth                      --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/selavi/  --resume --seed 100 > ./jobs/outs/selavi_full_finetune_out_$time.out

time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32_avid.yaml  --pretext-model-name  avid_cma      --pretext-model-path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar             --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/avid_cma/ --resume  --seed 100 > ./jobs/outs/avid_cma_full_finetune_out_$time.out


time=`date +%m-%d_%H-%M-%S`
/nfs/users/ext_malitha.gunawardhana/miniconda3/envs/severe_env1/bin/python finetune.py  configurations/benchmark/something/full_fine_tune/112x112x32.yaml       --pretext-model-name  gdt           --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth                                --finetune-ckpt-path ./checkpoints/ssv2/full_fine_tune/gdt/ --resume --seed 100 > ./jobs/outs/gdt_full_finetune_out_$time.out




