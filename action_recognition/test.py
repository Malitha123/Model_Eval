# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import tensor
import torch.multiprocessing as mp
import utils.logger
import yaml
from torch import nn
from utils import eval_utils, main_utils


# from utils.debug import debug


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_false')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--pretext-model-name', default='rspnet')
parser.add_argument('--pretext-model-path', default=None)
parser.add_argument('--finetune-ckpt-path', default='checkpoints/scratch/')

def distribute_model_to_cuda(model, args, cfg):
    if torch.cuda.device_count() == 1:
        model = model.cuda()
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        cfg['dataset']['batch_size'] = max(cfg['dataset']['batch_size'] // args.world_size, 1)
        cfg['num_workers'] = max(cfg['num_workers'] // args.world_size, 1)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model

def get_model( cfg, eval_dir, args, logger):

    from backbones import load_backbone

    ckp_manager = eval_utils.CheckpointManager(eval_dir, rank=args.gpu)

    model = load_backbone("r2plus1d_18", args.pretext_model_name,args.pretext_model_path)

    return model, ckp_manager


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, logger = eval_utils.prepare_environment(args, cfg, fold)

    # create pretext model
    model, ckp_manager = get_model( cfg, eval_dir, args, logger) 
    # modify last layer with specific number of classes
    model.fc = nn.Linear(cfg['model']['args']['feat_dim'], cfg['model']['args']['n_classes'])

    # Log model description
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(eval_utils.parameter_description(model))

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # Distribute
    model = distribute_model_to_cuda(model, args, cfg)

    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    ################################ Test only ################################
    if cfg['test_only']:
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_best=True)
        #start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
        logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.best_checkpoint_fn(), start_epoch))

    ################################ Eval ################################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    # cfg['dataset']['test']['clips_per_video'] = 5  # Evaluate clip-level predictions with 25 clips per video for metric stability
    cfg['dataset']['test']['clips_per_video'] = 25  # Evaluate clip-level predictions with 25 clips per video for metric stability
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    top1_dense, top5_dense, mean_top1, mean_top5 = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)

    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    #logger.add_line('Clip@1: {:6.2f}'.format(top1))
    #logger.add_line('Clip@5: {:6.2f}'.format(top5))

    logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
    logger.add_line('Video@MeanTop1: {:6.2f}'.format(mean_top1))
    logger.add_line('Video@5: {:6.2f}'.format(top5_dense))
    logger.add_line('Video@MeanTop5: {:6.2f}'.format(mean_top5))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.4e')
    top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    top5_meter = metrics_utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.logger.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, top1_meter, top5_meter],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')
    if phase in {'test_dense', 'test'}:
        model = eval_utils.BatchWrapper(model, cfg['dataset']['batch_size'])

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    end = time.time()
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))

    # collect all predictions and targets
    all_outputs = []
    all_targets = []
    incorrect_df = pd.DataFrame()
    correct_df = pd.DataFrame()
    correct_count_df = pd.DataFrame()
    incorrect_count_df = pd.DataFrame()
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['frames']
        target = sample['label'].cuda()

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            video = video.cuda()
        #print(video.size())
        # print('video', video.shape)
        # print('sample', target)
        # compute outputs
        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            labels_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, labels_tiled)
            #print(confidence.size(),labels_tiled.size())
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)
        #print(confidence.size(),target.size())
        
        all_outputs.append(confidence)
        all_targets.append(target)

        with torch.no_grad():
            # acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
            (acc1, acc5), incorrect_pred, correct_pred = metrics_utils.accuracy(confidence, target, topk=(1, 5))
            # print('incorrect_pred', incorrect_pred, type(incorrect_pred))
            incorrect_df = pd.concat([incorrect_df, pd.DataFrame(tensor(incorrect_pred).cpu().numpy())])
            correct_df   = pd.concat([correct_df, pd.DataFrame(tensor(correct_pred).cpu().numpy())])
            loss_meter.update(loss.item(), target.size(0))
            top1_meter.update(acc1[0], target.size(0))
            top5_meter.update(acc5[0], target.size(0))
        
        # print("------------------------------------------------")
        # print("incorrect_df",incorrect_df)
        # print("correct_df", correct_df)
        # print("#################################################")
        # print("incorrect_df val counts", incorrect_df.value_counts())
        # print("correct_df counts", correct_df.value_counts())

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)

    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    classes = all_targets.unique()

    classwise_top1 = [0 for c in classes]
    classwise_top5 = [0 for c in classes]
    for c in classes:
        indices = all_targets == c
        (mean_top1, mean_top5),  mean_incorrect, mean_correct = metrics_utils.accuracy(all_outputs[indices], all_targets[indices], topk=(1, 5))
        classwise_top1[c] = mean_top1
        classwise_top5[c] = mean_top5
    classwise_top1 = torch.cat(classwise_top1).mean()
    classwise_top5 = torch.cat(classwise_top5).mean()

    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size)
    
    # print("incorrect_df",incorrect_df)
    # print("correct_df", correct_df)
    # print("incorrect_df", incorrect_df.value_counts())
    # print("correct_df", correct_df.value_counts())
    incorrect_count = incorrect_df.value_counts()
    temp_id_incorrect = [list(t) for t in incorrect_count.index]
    temp_id_incorrect = np.array(temp_id_incorrect).flatten()
    count_incorrect = incorrect_count.values
    # plt.figure()
    # plt.bar(x,y)
    # plt.xlabel("video id")
    # plt.ylabel("count")
    # plt.title('incorrect predictions')
    # plt.savefig('incorrect_predictions.png')
    # plt.savefig(f'{args.pretext_model_name}_incorrect_predictions.png')


    correct_count = correct_df.value_counts()
    temp_id_correct = [list(t) for t in correct_count.index]
    temp_id_correct = np.array(temp_id_correct).flatten()
    count_correct = correct_count.values
    # plt.figure()
    # print("x",x)
    # print("y",y)
    # plt.bar(x,y)
    # plt.xlabel("video id")
    # plt.ylabel("count")
    # plt.title('correct predictions')
    # plt.savefig(f'{args.pretext_model_name}_correct_predictions.png')
    correct_count_df['correct_predicition_temp_id'] = temp_id_correct
    correct_count_df['correct_predicition_count'] = count_correct
    incorrect_count_df['incorrect_predicition_temp_id'] = temp_id_incorrect
    incorrect_count_df['incorrect_predicition_count'] = count_incorrect

    correct_count_df.to_csv(f'./jobs/results_file/{args.pretext_model_name}_correct_predicitions.csv',index=False)
    incorrect_count_df.to_csv(f'./jobs/results_file/{args.pretext_model_name}_incorrect_predicitions.csv',index=False)


    return top1_meter.avg, top5_meter.avg, classwise_top1, classwise_top5


if __name__ == '__main__':
    main()


