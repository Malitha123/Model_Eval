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
│   │   ├── linear


```

## Dataset Preparation

The datasets can be downloaded from the following links:

* [Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something)

* annoations that we use for each dataset in the ./data/ directory.
<!---
The expected directory hierarchy is as follow:-->
* We expect a directory hierarchy as below. After downloading the datasets from the original sources, please update the data and annotation paths for each dataset in the respective dataloader scripts e.g datasets/ucf.py, datasets/something.py, datasets/gym_99.py, etc. 
```
├── data
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
└── ...
```

## Experiments


* For finetuning pretrained models use the bash file of multi_job_full_fine_tune.sh

* For linear evaluation pretrained models use the bash file of multi_job_full_fine_tune.sh



This project is built upon [SEVERE-Benchmark](https://github.com/fmthoker/SEVERE-BENCHMARK/tree/main/action_recognition). Thanks to the contributors of these great codebases.
