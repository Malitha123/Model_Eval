# helper script to run sample training run
repo="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$repo

# get inputs from the user
while getopts "c:e:n:" OPTION; do
    case $OPTION in
        c) cfg=$OPTARG;;
		e) epoch=$OPTARG;;
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

# set epoch as the best checkpoint if it is not specified
if [ "$epoch" ==  ""  ]
then
    ckpt="checkpoint_best.pyth"
else
    file=$(printf %06d $epoch).pyth
    ckpt="checkpoint_epoch_$file"
fi


# output paths
expt_folder="$(basename -- $cfg)"
expt_folder="${expt_folder%.yaml}"
output_dir=/var/scratch/pbagad/expts/epic-kitchens-ssl/$expt_folder/
echo "Saving outputs: "$output_dir
mkdir -p $output_dir
logs_dir=$output_dir/logs/
mkdir -p $logs_dir

# checkpoint path
ckpt_path=$output_dir/checkpoints/$ckpt

echo ":::::::::::::::> Running eval for $cfg  :::::::::::::::"
echo "::::::: Checkpoint: $ckpt_path ::::::::"

if [ ! -f $ckpt_path ]; then
    echo ""
    echo "::::::: FAILED: Checkpoint not found at $ckpt_path!"
    exit
else
    echo "::::::: SUCCESS: Checkpoint file found! Running evaluation ..."
fi

# dataset paths
dataset_dir=/local-ssd/pbagad/EPIC-KITCHENS/
annotations_dir=/local-ssd/pbagad/datasets/EPIC-KITCHENS-100/annotations/

# run evaluation
python tools/run_net.py \
    --cfg $cfg \
    NUM_GPUS $num_gpus \
    OUTPUT_DIR $output_dir \
    EPICKITCHENS.VISUAL_DATA_DIR $dataset_dir \
    EPICKITCHENS.ANNOTATIONS_DIR $annotations_dir \
    TEST.CHECKPOINT_FILE_PATH $ckpt_path \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    > $logs_dir/val_logs_$ckpt.txt \
