"""Script to upload old runs to W&B via log file."""
from os.path import join, exists, basename, splitext, expanduser
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np

from tools.run_net import parse_args, load_config


def read_txt(path: str):
    with open(path, "rb") as f:
        lines = f.read()
        lines = lines.decode("utf-8")
        lines = lines.split("\n")
    return lines


def get_logged_stats(lines):
    stat_lines = []
    for line in lines:
        if "epoch" in line and "_type" in line:
            stat_lines.append(eval(line.split("[INFO: logging.py:   67]: json_stats: ")[-1]))

    return stat_lines


def init_wandb(name, project="video-ssl", entity="uva-vislab", dir=expanduser("~")):
    wandb.init(name=name, project=project, entity=entity, dir=dir)


def get_iter_count(ep, it):
    it, num_samples = it.split("/")
    it = int(it)
    num_samples = int(num_samples)


if __name__ == "__main__":

    # SlowFast
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/SLOWFAST/32x224x224_R50_K400_8x8.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x224x224_R50_K400_8x8/logs/train_logs.txt"

    # CTP Model
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/CTP/32x112x112_R2+1D-18_K400_LR0.01.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R2+1D-18_K400_LR0.01/logs/train_logs.txt"

    # CTP with LR = 0.0025, no warmup
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/CTP/32x112x112_R2+1D-18_K400_LR0.0025.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R2+1D-18_K400_LR0.0025/logs/train_logs.txt"

    # R2+1D with original hyperparameters (LR = 0.01)
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.01.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.01/logs/train_logs.txt"

    # R2+1D with LR = 0.0025
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0025/logs/train_logs.txt"

    # R2+1D with LR = 0.0025 with different scheduler
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025_SCHED-0-15-25.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0025_SCHED-0-15-25/logs/train_logs.txt"

    # R2+1D with LR = 1e-4
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0001.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0001/logs/train_logs.txt"

    # R2+1D with LR = 1e-3
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.001.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.001/logs/train_logs.txt"

    # R2+1D with LR 0.0025 and consecutive 32 frames
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_SR_1_R18_K400_LR0.0025.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_SR_1_R18_K400_LR0.0025/logs/train_logs.txt"

    # R2+1D with LR 0.0025 and temporal resolution 4 (32 out of 128 frames are chosen in a clip)
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_SR_4_R18_K400_LR0.0025.yaml"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_SR_4_R18_K400_LR0.0025/logs/train_logs.txt"

    # GDT
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/GDT/32x112x112_GDT_R2+1D_K400_LR0.0025.yaml"    
    # log_path = "/var/scratch/pbagad/expts/epic-kitchens-ssl/32x112x112_GDT_R2+1D_K400_LR0.0025/logs/train_logs.txt"
   
    # Prextext contrast
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/PRETEXT_CONTRAST/32x112x112_PC_R18_K400_LR0.0025.yaml"
    # log_path = "/var/scratch/pbagad/expts/epic-kitchens-ssl/32x112x112_PC_R18_K400_LR0.0025/logs/train_logs.txt"
 
    # RPSNet
    cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/RSPNET/32x112x112_R18_K400_LR0.0025.yaml"
    log_path = "/var/scratch/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0025/logs/train_logs.txt"

    # load cfg
    args = parse_args()
    args.cfg_file = cfg_path
    cfg = load_config(args)

    lines = read_txt(log_path)
    stat_lines = get_logged_stats(lines)
    df = pd.DataFrame(stat_lines)

    # add column for epochs
    df["epoch"] = df["epoch"].apply(lambda x: x.split("/")[0]).astype(int)

    # add column for iterations
    train_total_iter = 0
    val_total_iter = 0
    for i in range(len(df)):
        iter = df.iloc[i]["iter"]
        _type = df.iloc[i]["_type"]
        if isinstance(iter, str):
            if "train" in _type:
                train_total_iter += int(iter.split("/")[0])
                df.at[i, "step"] = int(train_total_iter)
            if "val" in _type:
                val_total_iter += int(iter.split("/")[0])
                df.at[i, "step"] = int(val_total_iter)

    # df["step"] = df[["epoch", "iter"]].apply(lambda x: x[1]., axis=1)
    df = df.dropna(subset=["step"])
    df.step = df.step.astype(int)
    steps = df.step.values

    losses = [x for x in df.columns if "loss" in x]
    metrics = [x for x in df.columns if "_acc" in x]

    # initialize W&B
    run_name = cfg_path.split("epic-kitchens-ssl/configs/")[-1]
    run_name = run_name.replace("/", " | ")
    init_wandb(name=run_name, entity="uva-vislab", project="video-ssl")

    # upload config
    config = wandb.config
    config.config_name = basename(cfg_path)
    config.update(cfg)

    # log train stats
    train_df = df[df["_type"].apply(lambda x: "train" in x)]
    train_df = train_df.rename(columns={k: "train/" + k for k in (metrics + losses + ["epoch", "step"])})
    train_records = train_df.to_dict('records')
    for r in tqdm(train_records, desc="Logging train stats"):
        wandb.log(r)

    # log val stats
    val_df = df[df["_type"].apply(lambda x: "val" in x)]
    val_df = val_df.rename(columns={k: "val/" + k for k in (metrics + losses + ["epoch", "step"])})
    val_records = val_df.to_dict('records')
    for r in tqdm(val_records, desc="Logging val stats"):
        wandb.log(r)

