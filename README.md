## Setup

We recommend creating a `conda` environment and installing dependencies in it by using:
```bash
conda create -n model_eval_env python=3.7
conda activate model_eval_env
pip install -r requirements.txt 

```

We run our experiments on Python 3.7 and PyTorch 1.6. 


## Evaluated models

* you need the pre-trained checkpoints for each method at  path `../checkpoints_pretraining/`

* Download pretrain weights for each method from [here](https://surfdrive.surf.nl/files/index.php/s/Zw9tbuOYAInzVQC) and unzip it. It contains Kinetics-400 pretrained R(2+1D)-18 weights

* Create a 'jobs' folder and subdirectories as follows to run slurms files. 
```
├── jobs
│   ├──csvs
│   ├──full_fine_tune_evaluation_results
│   ├──linear_fine_tune_results
│   ├──outs
│   ├──slurms

* Create a 'checkpoints' folder and subdirectories as follows to save checkpoints.

├── checkpoints
│   ├──ssv2
│   │   ├── full_fine_tune
│   │   ├── direct
│   │   ├── linear
│   ├──ntu60
│   │   ├── full_fine_tune
│   │   ├── direct
│   │   ├── linear

```

## Dataset Preparation

The datasets can be downloaded from the following links:

* [UCF101 ](http://crcv.ucf.edu/data/UCF101.php)
* [Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
* For [Fine-Gym v_1.0](https://sdolivia.github.io/FineGym/) please send a request to Fine-Gym authors via [Form](https://docs.google.com/forms/d/e/1FAIpQLScg8KDmBl0oKc7FbBedT0UJJHxpBHQmgsKpc4nWo4dwdVJi0A/viewform) to get access to the dataset. After downloading the videos, follow  the script provided in [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) to trim the videos to subactions. (Note, if you also dowload the videos via [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md) script some of the video will be  missing because of the broken youtube links, so we suggest to request the Fine-Gym authors to get access to whole dataset). Please contact us in case of any issues or if you need preprocessed the Fine-Gym videos. 

* annoations that we use for each dataset in the ./data/ directory.
<!---
The expected directory hierarchy is as follow:-->
* We expect a directory hierarchy as below. After downloading the datasets from the original sources, please update the data and annotation paths for each dataset in the respective dataloader scripts e.g datasets/ucf.py, datasets/something.py, datasets/gym_99.py, etc. 
```
├── data
│   ├──ucf101
│   │   ├── ucfTrainTestlist
│   │   │   ├── classInd.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── trainlist01.txt
│   │   │   └── ...
│   │   └── UCF-101
│   │       ├── ApplyEyeMakeup
│   │       │   └── *.avi
│   │       └── ...
│   ├──gym
│   │   ├── annotations
│   │   │   ├── gym99_train.txt
│   │   │   ├── gym99_val.txt 
│   │   │   ├── gym288_train.txt
│   │   │   ├── gym288_val.txt
│   │   │   └──
│   │   └── videos
│   │       ├── *.avi
│   │       └── ...
│   │
│   │──smth-smth-v2
│   │   ├── something-something-v2-annotations
│   │   │   ├── something-something-v2-labels.json
│   │   │   ├── something-something-v2-test.json
│   │   │   ├── something-something-v2-train.json
│   │   │   └── something-something-v2-validation.json
│   │   │       
│   │   └── something-something-v2-videos_avi
│   │       └── *.avi
│   │          
│   ├──ntu60
│   │   ├── ntu_60_cross_subject_TrainTestlist
│   │   │   ├── classInd.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── trainlist01.txt
│   │   │   └── ...
│   │   └── videos
│   │       ├── brushing_hair
│   │       │   └── *.avi
│   │       ├── brushing_teeth
│   │       │   └── *.avi
│   │       └── ...
│   │
│   ├──kinetics-400
│   │   ├── labels
│   │   │   ├── train_videofolder.txt
│   │   │   ├── val_videofolder.txt
│   │   │   └── ...
│   │   └── VideoData
│   │       ├── playing_cards
│   │       │   └── *.avi
│   │       ├── singing
│   │       │   └── *.avi
│   │       └── ...
└── ...
```

## Experiments


* For finetuning pretrained models use the bash file of train_full_fine_tune.sh. Then run the evaluation_full_finetune.sh

* For linear evaluation pretrained models use the bash file of train_linear_fine_tune.sh. Then run the evaluation_linear.sh

* For direct evaluation pretrained models use the bash file of evaluation_direct.sh
