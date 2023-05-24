"""Reads results of the following experiments.

Experiment: Ablation on number of training samples.
Dataset: EPIC-KITCHENS-100.
"""

import subprocess
from os.path import join, exists
from glob import glob
import re
import ast
from natsort import natsorted


def get_hostinfo(filter_by="science.uva.nl"):
    """Get host information."""
    hostname = socket.gethostname()
    outputs = subprocess.check_output(
        [ 'hostname', '--all-fqdns' ]
    ).decode("utf-8").strip("\n").split(" ")
    fqdn = [ x for x in outputs if x.endswith(filter_by) ][0]
    fqdn = fqdn.split(filter_by)[0][:-1]
    
    hostinfo = {
        "hostname": hostname,
        "fqdn": fqdn,
    }
    return hostinfo


def modify_model_name(model, scratch=False):
    """Modify the model name."""
    if model == "VIDEOMOCO":
        return "Video Moco"
    elif model == "R2PLUS1D":
        if not scratch:
            return "Supervised pretraining"
        else:
            return "No pretraining"
    elif model == "PRETEXT_CONTRAST":
        return "Pretext-Contrast"
    elif model in ["SELAVI", "CTP", "TCLR", "RSPNET", "GDT"]:
        return model
    elif model == "MOCO":
        return "MoCo"
    elif model == "AVID_CMA":
        return "AVID-CMA"

    return model


if __name__ == "__main__":
    import argparse
    import socket
    
    # hostname = socket.gethostname()
    
    dataset_name = "EK-100"
    
    hostinfo = get_hostinfo()
    hostname = hostinfo["fqdn"]

    hostname_to_outdir = {
        "diva": "/home/pbagad/expts/epic-kitchens-ssl",
        "fs4.das5": "/var/scratch/pbagad/expts/epic-kitchens-ssl",
        "fs4.das6": "/var/scratch/pbagad/expts/epic-kitchens-ssl",
    }

    parser = argparse.ArgumentParser(
        description="Arguments for experiment info."
    )
    parser.add_argument(
        "--device",
        help="Device where the results are stored.",
        default=hostname,
        type=str,
    )
    parser.add_argument(
        "--outdir",
        help="Output directory where the results are stored.",
        default=hostname_to_outdir[hostname],
        type=str,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="VSSL model used.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--key",
        "-k",
        help="Key to show results for. For e.g., `verb_top1_acc`",
        default="verb_top1_acc",
        type=str,
    )
    parser.add_argument(
        "--search_prefix",
        "-s",
        help="Prefix in expt folder name.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--scratch",
        help="R2PLUS1D trained from scratch.",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    
    model = (args.model).upper()
    model_outdirs = glob(join(args.outdir, model + "--*n_samples_*_*"))

    if args.scratch:
        model_outdirs = [ x for x in model_outdirs if "_scratch_" in x ]
    import ipdb; ipdb.set_trace()

    model_base_outdir = model_outdirs[0]
    model_base_outdir = re.sub(args.search_prefix + "n_samples_\d+_", "", model_base_outdir)

    scratch = "_scratch_" in model_outdirs[0]
    print(scratch, model_outdirs[0])
    val_log_files = [join(x, "logs/val_logs_checkpoint_best.pyth.txt") for x in model_outdirs]
    val_log_files = natsorted(val_log_files)
    val_log_files += [join(model_base_outdir, "logs/val_logs_checkpoint_best.pyth.txt")]
        
    pattern =  args.search_prefix + "n_samples_\d{4,5}_"
    num_samples = [re.search(pattern, x).group() for x in model_outdirs]
    num_samples = natsorted([int(x.split("_")[-2]) for x in num_samples])
    num_samples += ["Full"]
    
    display = []
    for n, f in zip(num_samples, val_log_files):

        assert exists(f), "File {} does not exist.".format(f)
        
        lines = open(f).readlines()
        results =  ast.literal_eval(lines[-1].strip("\n"))
        
        assert results["split"] == "test_final", "Split is not test_final."
        assert args.key in results, "Key {} not found.".format(args.key)
        
        to_show = [modify_model_name(model, scratch), dataset_name, n, results[args.key]]
        print("{},{},{},{}".format(*to_show))
