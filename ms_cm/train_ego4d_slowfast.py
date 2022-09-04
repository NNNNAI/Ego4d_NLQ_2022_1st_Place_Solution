import os
import time
import json
import pprint
import random
import copy
import logging
import numpy as np
import ms_cm.vslnet_utils.evaluate_ego4d_nlq as ego4d_eval

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ms_cm.configs import BaseOptions
from ms_cm.sw_vs_ego4d_dataset import \
    Ego4d_dataset, Ego4d_collate, prepare_batch_inputs
from ms_cm.inference_ego4d_slowfast import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from ms_cm.vslnet_utils.data_gen_ego4d import gen_or_load_dataset
from ms_cm.vslnet_utils.runner_utils import filter_checkpoints,eval_test
from easydict import EasyDict as edict

from ms_cm.vslnet_utils.data_util import (
    load_json,
    load_lines,
    load_pickle,
    save_pickle,
    time_to_index,
)
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
def extract_index(start_logits, end_logits):
    start_prob = nn.Softmax(dim=1)(start_logits)
    end_prob = nn.Softmax(dim=1)(end_logits)
    outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
    outer = torch.triu(outer, diagonal=0)

    # Get top 5 start and end indices.
    batch_size, height, width = outer.shape
    outer_flat = outer.view(batch_size, -1)
    _, flat_indices = outer_flat.topk(5, dim=-1)
    start_indices = flat_indices // width
    end_indices = flat_indices % width
    return start_indices, end_indices

def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer,gt_json_path='../../Datasets/Ego4d/ego4d_annotation/nlq_train.json'):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    predictions = []
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        query_record = batch[0]
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory,cur_scale=batch[2])
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        outputs = model(**model_inputs)


        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
    # print(opt.results_dir)
    best_metric = -1.0
    score_writer = open(
            os.path.join(opt.results_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    eval_loader = DataLoader(
        val_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            if opt.optim_name == 'AdamW':
                lr_scheduler.step()
        eval_epoch_interval = 1
        if  (epoch_i + 1) % eval_epoch_interval == 0:
            print(
                f"\nEpoch: {epoch_i + 1:2d} |", flush=True
            )
            result_save_path = os.path.join(
                opt.results_dir,
                f"{epoch_i}_preds.json",
            )
            model.eval()
            with torch.no_grad():
                results, mIoU, score_str  = \
                    eval_epoch(model, eval_loader, opt, result_save_path,opt.eval_gt_json, epoch_i, tb_writer)

            print(score_str, flush=True)
            score_writer.write(score_str)
            score_writer.flush()

            if opt.optim_name == 'AdamW':
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
            elif opt.optim_name == 'BertAdam':
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
            print("_{:0>4d}.ckpt".format(epoch_i))
            torch.save(
                checkpoint,
                opt.ckpt_filepath.replace(".ckpt", "_{:0>4d}.ckpt".format(epoch_i))
            )
            # keep all checkpoints
            filter_checkpoints(opt.results_dir, suffix="ckpt", max_to_keep=opt.n_epoch)

    tb_writer.close()
    score_writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    dataset_config = dict(
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        multiscale_list=opt.numscale_list,
        use_sw=opt.use_sw,
        sw_len_ratio=opt.sw_len_ratio,
        use_vs=opt.use_vs,
        vs_prob=opt.vs_prob,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        txt_drop_ratio=opt.txt_drop_ratio,
    )
    vslnet_datasetconfigs = edict({'task':opt.vsldataset_task, 'fv':opt.vsldataset_fv, 
            'max_pos_len': opt.max_v_l,'num_workers':opt.vsldataset_num_workers,\
                'vslnet_datapath':opt.vslnet_datapath,'save_dir':opt.vslnet_dataset_save_dir,
                'thres_in_train':opt.vslnet_thres_in_train})
    # preparing ego4d dataset by using vslnet configuration
    dataset = gen_or_load_dataset(vslnet_datasetconfigs)

    dataset_config['dataset'] = dataset["train_set"]
    dataset_config['mode'] = 'train'
    train_dataset = Ego4d_dataset(**dataset_config)

    dataset_config['dataset'] = dataset["val_set"]
    dataset_config['mode'] = 'eval'
    dataset_config["txt_drop_ratio"] = 0
    eval_dataset = Ego4d_dataset(**dataset_config)

    
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)


if __name__ == '__main__':
    start_training()
