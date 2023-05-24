# helper script to run sample training run
repo="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$repo

# get inputs from the user
while getopts "c:n:" OPTION; do
    case $OPTION in
        c) cfg=$OPTARG;;
        n) num_gpus=$OPTARG;;
        *) exit 1 ;;
    esac
done

# check cfg is given
if [ "$cfg" ==  "" ];then
       echo "cfg is a required argument; Please use -c <relative path to config> to pass config file."
       echo "You can choose configs from:"
       ls $repo/configs/*
       exit
fi

# set number of GPUs as 4 if not specified
if [ "$num_gpus" ==  "" ];then
       num_gpus=4
fi

# train_ckpt_path=/home/pbagad/expts/epic-kitchens-ssl/pretrained/SLOWFAST_8x8_R50.pkl

echo ":::::::::::::::> Running training for $cfg  :::::::::::::::"

# output paths
# expt_folder="$(basename -- $(basename -- $cfg))"
# expt_folder="${expt_folder%.yaml}"
expt_folder="${cfg%.yaml}"
IFS='/' read -r -a array <<< $expt_folder
expt_folder="${array[-2]}--${array[-1]}"

output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
echo "Saving outputs: "$output_dir
mkdir -p $output_dir
logs_dir=$output_dir/logs/
mkdir -p $logs_dir

# dataset paths
dataset_dir=/local/pbagad/EPIC-KITCHENS/
annotations_dir=/local/pbagad/datasets/EPIC-KITCHENS-100/annotations/

# run training
python tools/run_net.py \
    --cfg $cfg \
    --init_method tcp://localhost:9998 \
    NUM_GPUS $num_gpus \
    OUTPUT_DIR $output_dir \
    EPICKITCHENS.VISUAL_DATA_DIR $dataset_dir \
    EPICKITCHENS.ANNOTATIONS_DIR $annotations_dir > $logs_dir/train_logs.txt \
    # TRAIN.CHECKPOINT_FILE_PATH $train_ckpt_path
