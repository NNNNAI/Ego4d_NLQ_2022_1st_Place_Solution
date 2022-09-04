import pprint
from tqdm import tqdm, trange
import numpy as np
import os
import copy
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import ms_cm.vslnet_utils.evaluate_ego4d_nlq as ego4d_eval

from torch.utils.data import DataLoader

from easydict import EasyDict as edict
from ms_cm.model import build_model
from ms_cm.configs import TestOptions
from utils.basic_utils import save_jsonl, save_json
from ms_cm.vslnet_utils.data_gen_ego4d import gen_or_load_dataset
from ms_cm.pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from ms_cm.sw_vs_ego4d_dataset import Ego4d_dataset, Ego4d_collate, prepare_batch_inputs

import json
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)



    
def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time

@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, result_save_path,gt_json_path, epoch_i=None, tb_writer=None):
    model.eval()
    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None
    numscale_list = opt.numscale_list
    mr_res = []
    predictions = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_record = batch[0]
        start_indices_list = []
        end_indices_list = []
        faltten_score_list = []
        for scale_tmp in numscale_list:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory,cur_scale=scale_tmp)
            outputs = model(**model_inputs)

            start_indices, end_indices, faltten_scores = extract_index(outputs['start_logits'], outputs['end_logits'], opt.allscaletopk)
            start_indices_list.append(start_indices)
            end_indices_list.append(end_indices)
            faltten_score_list.append(faltten_scores)

        start_indices_total = torch.cat(start_indices_list,dim=1)
        end_indices_total = torch.cat(end_indices_list,dim=1)
        faltten_score_total = torch.cat(faltten_score_list,dim=1)


        _, flat_total_indices = faltten_score_total.topk(5, dim=-1)
        start_indices = torch.gather(start_indices_total, dim=1, index=flat_total_indices).cpu().numpy()
        end_indices = torch.gather(end_indices_total, dim=1, index=flat_total_indices).cpu().numpy()

        for record, starts, ends in zip(query_record, start_indices, end_indices):
            # Convert all indices to times.
            timewindow_predictions = []
            for start, end in zip(starts, ends):
                start_time, end_time = index_to_time(
                    start, end, record["v_len"], record["duration"]
                )
                timewindow_predictions.append([float(start_time), float(end_time)])
            new_datum = {
                "clip_uid": record["vid"],
                "annotation_uid": record["annotation_uid"],
                "query_idx": int(record["query_idx"]),
                "predicted_times": copy.deepcopy(timewindow_predictions),
            }
            predictions.append(new_datum)

    # Save predictions if path is provided.
    if result_save_path:
        with open(result_save_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": predictions,
                }, file_id
            )

    # Evaluate if ground truth JSON file is provided.
    if gt_json_path:
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        title = f"Epoch {epoch_i}"
        score_str = ego4d_eval.display_results_addmeanr1(
            results, mIoU, thresholds, topK, title=title
        )
    else:
        results = None
        mIoU = None
        score_str = None
    return results, mIoU, score_str



def get_eval_res(model, eval_loader, opt, result_save_path,gt_json_path, epoch_i, tb_writer):
    """compute and save query and video proposal embeddings"""
    results, mIoU, score_str = compute_mr_results(model, eval_loader, opt, result_save_path,gt_json_path, epoch_i, tb_writer)  # list(dict)
    return results, mIoU, score_str 

def extract_index(start_logits, end_logits,topk):
    start_prob = nn.Softmax(dim=1)(start_logits)
    end_prob = nn.Softmax(dim=1)(end_logits)
    outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
    outer = torch.triu(outer, diagonal=0)

    # Get top 5 start and end indices.
    batch_size, height, width = outer.shape
    outer_flat = outer.view(batch_size, -1)
    outer_flat_score, flat_indices = outer_flat.topk(5, dim=-1)
    start_indices = flat_indices // width
    end_indices = flat_indices % width
    return start_indices, end_indices,outer_flat_score

def eval_epoch(model, eval_loader, opt, result_save_path,gt_json_path, epoch_i=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()

    results, mIoU, score_str = get_eval_res(model, eval_loader, opt, result_save_path,gt_json_path, epoch_i, tb_writer)
    return results, mIoU, score_str 

def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    if opt.optim_name == 'AdamW':
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
        optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    elif opt.optim_name == 'BertAdam':
        num_train_optimization_steps = int(opt.ego4d_train_num / opt.bsz) * opt.n_epoch
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": opt.wd,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay  )and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=opt.lr,
                warmup=opt.warmup_proportion,
                t_total=num_train_optimization_steps,
        )
        lr_scheduler = None


    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if opt.optim_name == 'AdamW':
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    # You should change below param with your our datapath
    opt.vslnet_datapath = '../episodic-memory/NLQ/VSLNet/data/'
    eval_gt_json = '../episodic-memory/NLQ/VSLNet/data/nlq_val.json'
    train_gt_json = '../../Datasets/Ego4d/ego4d_annotation/nlq_train.json'
    opt.v_feat_dirs = ['../episodic-memory/NLQ/VSLNet/data/features/nlq_official_v1/official',\
                        './CLIP_feature_slowfast/Slowfast_vitb16CLIP_videofeature']

    
    vslnet_datasetconfigs = edict({'task':opt.vsldataset_task, 'fv':opt.vsldataset_fv, 
            'max_pos_len': opt.max_v_l,'num_workers':opt.vsldataset_num_workers,\
                'vslnet_datapath':opt.vslnet_datapath,'save_dir':opt.vslnet_dataset_save_dir,\
                'thres_in_train':opt.vslnet_thres_in_train})
    # preparing ego4d dataset by using vslnet configuration
    dataset = gen_or_load_dataset(vslnet_datasetconfigs)


    cur_epoch = opt.resume.split('model_')[-1].split('.ckpt')[0]
    print(cur_epoch)
    if opt.split =='train':
        dataset_name = 'train_set'
        json_name =  f"{cur_epoch}_train_preds.json"
        gt_json = train_gt_json
    elif opt.split =='val':
        dataset_name = 'val_set'
        json_name =  f"{cur_epoch}_val_preds.json"
        gt_json = eval_gt_json
    elif opt.split =='test':
        dataset_name = 'test_set'
        json_name =  f"{cur_epoch}_test_preds.json"
        gt_json = None

    eval_dataset = Ego4d_dataset(
        dataset=dataset[dataset_name],
        mode='eval',
        use_sw=opt.use_sw,
        sw_len_ratio=opt.sw_len_ratio,
        use_vs=opt.use_vs,
        vs_prob=opt.vs_prob,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        multiscale_list=opt.numscale_list,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        # data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        txt_drop_ratio=0
    )

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    model, criterion, _, _ = setup_model(opt)

    save_dir = os.path.join(opt.results_dir,'preds')
    os.makedirs(save_dir,exist_ok = True)

    result_save_path = os.path.join(
                save_dir,
                json_name
            )
    logger.info("Starting inference...")
    with torch.no_grad():
        results, mIoU, score_str =eval_epoch(model, eval_loader, opt, result_save_path,gt_json, None, None)

    print(score_str, flush=True)


if __name__ == '__main__':
    start_inference()
