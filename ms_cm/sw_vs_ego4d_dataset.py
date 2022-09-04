import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d,pad_sequences_1d_video_numscale
from ms_cm.vslnet_utils.data_util import time_to_index
import torch.nn.functional as F
import time
import os
from random import choice
import math
import random
import copy
from random import randint

logger = logging.getLogger(__name__)

# slowfast time_unit
# slowfast stride is 16 frame, so for 30-fps, the time_uint = 16/30 = 0.5333333333333333
time_unit = 0.5333333333333333
class Ego4d_dataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dataset, mode, v_feat_dirs, q_feat_dir,
                 multiscale_list,use_sw,sw_len_ratio,
                 use_vs,vs_prob,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True, txt_drop_ratio=0):

        self.dataset = dataset 
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.txt_drop_ratio = txt_drop_ratio
        self.mode = mode
        self.multiscale_list = multiscale_list

        self.use_sw = use_sw
        self.sw_len_ratio_start, self.sw_len_ratio_end = sw_len_ratio
        self.vs_prob = vs_prob
        self.use_vs = use_vs

        if self.mode == 'eval' or self.mode == 'test':
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        random_index = randint(0,len(self.dataset)-1)
        record = copy.deepcopy(self.dataset[index])
        record_random = copy.deepcopy(self.dataset[random_index])

        s_time, e_time = record["exact_s_time"], record["exact_e_time"]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        duration = record["duration"]
        vfeat_len = record["v_len"]
        self.record_vid = record["vid"]
        self.record_query = record["query"] 
        record['multiscale_list'] = self.multiscale_list

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(record['annotation_uid'],record['query_idx'])  # (Dq, ) or (Lq, Dq)
                                          
        if self.use_video:
            model_inputs["video_feat"], record["v_len"],s_ind, e_ind = self._get_video_feat_by_vid(record,record_random)  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_sw and self.mode == 'train':
            record["s_ind"], record["e_ind"] = s_ind, e_ind 

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            if self.mode == 'eval' or self.mode == 'test':
                pass 
            elif self.mode == 'train':
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(s_ind, e_ind, ctx_l)  # only one gt
        return dict(record=record, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, s_ind, e_ind, ctx_l, max_n=2):
        gt_st = int(s_ind)
        gt_ed = max(0, min(int(e_ind), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        # neg_clip_indices = random.sample(neg_pool, k=max_n)
        try :
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = [0,0]
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """
        Due to Ego4d NLQ dataset do not have saliency_labels, therefore we modify this function from moment_detr
        Sum the scores from annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices


    def _get_query_feat_by_qid(self, annotation_uid,query_idx):
        q_feature = torch.load(os.path.join(self.q_feat_dir, '{}_{}.pt'.format(annotation_uid,query_idx)))
        q_feat = q_feature.numpy().astype(np.float32)
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, record,record_random):
        vid = record["vid"]
        random_vid = record_random["vid"]
        gt_s_time, gt_e_time = record["exact_s_time"], record["exact_e_time"]

        if self.use_sw and self.mode == 'train':
            cur_sw_len_ratio = random.uniform(self.sw_len_ratio_start, self.sw_len_ratio_end)

        rand_s_idx = None
        v_feat_list = []
        for idx,_feat_dir in enumerate(self.v_feat_dirs):
            _feat_path = join(_feat_dir, f"{vid}.pt")
            _feat = torch.load(_feat_path).numpy().astype(np.float32)

            # Variable-length Sliding Window Sampling (SW)
            if self.use_sw and self.mode == 'train':
                if idx ==0:
                    total_feat_len = _feat.shape[0]
                    s_ind, e_ind = math.floor(min(gt_s_time/record["duration"],1.0)*(total_feat_len-1)), math.ceil(min(gt_e_time/record["duration"],1.0)*(total_feat_len-1))
                    ori_s_ind, ori_e_ind = s_ind, e_ind
                    total_feat_len_mulanchor = int(total_feat_len * cur_sw_len_ratio)
                    ori_clip_len = e_ind-s_ind+1
                    if total_feat_len_mulanchor <= ori_clip_len:
                        if ori_clip_len == total_feat_len:
                            total_feat_len_mulanchor = total_feat_len
                        else:
                            total_feat_len_mulanchor = random.randrange(ori_clip_len,total_feat_len)
                    rand_start_choice = max(0, e_ind-total_feat_len_mulanchor)
                    rand_end_choice = min(s_ind, total_feat_len - total_feat_len_mulanchor)

                    rand_s_idx = random.randrange(rand_start_choice,rand_end_choice) \
                        if total_feat_len > total_feat_len_mulanchor and rand_start_choice <rand_end_choice else rand_end_choice
                    rand_e_idx = min(rand_s_idx + total_feat_len_mulanchor, total_feat_len)
                    
                    gt_s_time = gt_s_time - rand_s_idx*time_unit
                    gt_e_time = gt_e_time - rand_s_idx*time_unit

                _feat = _feat[rand_s_idx:rand_e_idx]
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        v_feat = np.concatenate(v_feat_list, axis=1)

        # Video Splicing (VS)
        random_prob = random.random()
        cur_vs = False
        if self.use_vs and self.mode == 'train' and random_prob < self.vs_prob:
            cur_vs = True

        if cur_vs:
            total_feat_len = v_feat.shape[0]

            random_v_feat_list = []
            for idx,_feat_dir in enumerate(self.v_feat_dirs):
                _feat_path = join(_feat_dir, f"{random_vid}.pt")
                _feat = torch.load(_feat_path).numpy().astype(np.float32)
                if self.normalize_v:
                    # _feat = F.normalize(_feat, p=2.0, dim=-1)
                    _feat = l2_normalize_np_array(_feat)
                random_v_feat_list.append(_feat)
            random_v_feat = np.concatenate(random_v_feat_list, axis=1)
            random_v_len = random_v_feat.shape[0]
            random_v_takein_index = randint(1,random_v_len-2)

            final_v_list = []
            final_v_list.append(random_v_feat[:random_v_takein_index])

            final_v_list.append(v_feat)

            final_v_list.append(random_v_feat[random_v_takein_index:])

            v_feat = np.concatenate(final_v_list, axis=0)

            gt_s_time = gt_s_time + random_v_takein_index*time_unit
            gt_e_time = gt_e_time + random_v_takein_index*time_unit


        pre_v_feat_len = v_feat.shape[0]
        v_feat = visual_feature_sampling(
                v_feat, max_num_clips=self.max_v_l
            )
        vlen = v_feat.shape[0]
        if self.use_sw and self.mode == 'train':
            s_ind, e_ind = math.floor(min(gt_s_time/(pre_v_feat_len*time_unit),1.0)*(vlen-1)), math.ceil(min(gt_e_time/(pre_v_feat_len*time_unit),1.0)*(vlen-1))
            assert e_ind<vlen,'e_ind {}, vlen {}'.format(e_ind, vlen)
        else:
            s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        return torch.from_numpy(v_feat), vlen, s_ind, e_ind  # (Lv, D)
        


def Ego4d_collate(batch):
    # collate_start_time = time.time()
    batch_record = [e["record"] for e in batch]  # seems no need to collate ?

    multiscale_list = batch_record[0]['multiscale_list']
    # print(multiscale_list)
    cur_scale = choice(multiscale_list)
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()

    for k in model_inputs_keys:
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == 'video_feat':
            batched_data[k] = pad_sequences_1d_video_numscale(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None,cur_scale=cur_scale)
        else:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    # get highlight label

    batch_v_len = [record["v_len"] for record in batch_record]
    batch_size = len(batch_v_len)
    max_len = max(batch_v_len)
    batch_s_ind = [int(record["s_ind"]) for record in batch_record]
    batch_e_ind = [int(record["e_ind"]) for record in batch_record]

    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = batch_s_ind[idx], batch_e_ind[idx]
        cur_max_len = batch_v_len[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_ : (et_ + 1)] = 1
        else:
            h_labels[idx][st : (et + 1)] = 1

    h_labels = torch.tensor(h_labels, dtype=torch.int64)

    NPM_labels = np.zeros(shape=[batch_size, cur_scale], dtype=np.int32)
    split_len = math.ceil(max_len /cur_scale)
    end_point_list = []
    start_point_list = []
    for i in range(cur_scale):
        if i != cur_scale-1:
            end_point_list.append((i+1)*split_len)
        else:
            end_point_list.append(max_len)
        start_point_list.append(i*split_len)
    for idx in range(batch_size):
        st, et = batch_s_ind[idx], batch_e_ind[idx]
        for i in range(cur_scale):
            if (st>=start_point_list[i] and st<end_point_list[i]) or (et>=start_point_list[i] and et<end_point_list[i]):
                NPM_labels[idx][i] = 1
            

    NPM_labels = torch.tensor(NPM_labels, dtype=torch.int64)

    batched_data['highlight_label']  = h_labels
    batched_data['NPM_labels']  = NPM_labels
    batched_data['start_label']  = torch.tensor(batch_s_ind, dtype=torch.int64)
    batched_data['end_label']  = torch.tensor(batch_e_ind, dtype=torch.int64)
    return batch_record, batched_data,cur_scale


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False,cur_scale=1):

    video_len = batched_model_inputs["video_feat"][0].shape[1]
    split_len = math.ceil(video_len/cur_scale)

    src_vid_split = torch.split(batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),split_len,dim=1)
    src_vid_mask_split = torch.split(batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),split_len,dim=1)
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid_total=src_vid_split,
        src_vid_mask_total=src_vid_mask_split,
        scale_mask=batched_model_inputs["video_feat"][2].to(device, non_blocking=non_blocking)
    )
    targets = {}
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    targets['highlight_label'] = batched_model_inputs['highlight_label'].to(device, non_blocking=non_blocking)
    targets['NPM_labels'] = batched_model_inputs['NPM_labels'].to(device, non_blocking=non_blocking)
    targets['start_label'] = batched_model_inputs['start_label'].to(device, non_blocking=non_blocking)
    targets['end_label'] = batched_model_inputs['end_label'].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets


def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature