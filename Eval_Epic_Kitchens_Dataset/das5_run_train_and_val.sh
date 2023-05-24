# setup env
conda activate slowfast-gpu

# echo ":: CTP"
# cfg=configs/EPIC-KITCHENS/CTP/das5_32x112x112_latest_ckpt_R2+1D-18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: PRETEXT-CONTRAST"
# cfg=configs/EPIC-KITCHENS/PRETEXT_CONTRAST/das5_32x112x112_PC_R18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: FULLY SUPERVISED"
# cfg=configs/EPIC-KITCHENS/R2PLUS1D/das5_32x112x112_R18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: RSPNET_Snellius"
# cfg=configs/EPIC-KITCHENS/RSPNET/das5_32x112x112_R18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: SELAVI"
# cfg=configs/EPIC-KITCHENS/SELAVI/das5_32x112x112_R18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: TCLR"
# cfg=configs/EPIC-KITCHENS/TCLR/das5_32x112x112_R18_K400_LR0.0025_no_norm_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt


# echo ":: VIDEOMOCO"
# cfg=configs/EPIC-KITCHENS/VIDEOMOCO/das5_32x112x112_R18_K400_LR0.0025_linear.yaml
# bash das5_train.sh -c $cfg
# bash das5_val.sh -c $cfg

# expt_folder="${cfg%.yaml}"
# IFS='/' read -r -a array <<< $expt_folder
# expt_folder="${array[-2]}--${array[-1]}"
# output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt

echo ":: COCLR"
cfg=configs/EPIC-KITCHENS/COCLR/das5_32x112x112_R18_K400_LR0.0025_linear.yaml
bash das5_train.sh -c $cfg
bash das5_val.sh -c $cfg

expt_folder="${cfg%.yaml}"
IFS='/' read -r -a array <<< $expt_folder
expt_folder="${array[-2]}--${array[-1]}"
output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
tail $output_dir/logs/val_logs_checkpoint_best.pyth.txt

