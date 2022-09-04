"""
Multi-scale Cross-modal Transformer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn


from ms_cm.transformer import build_transformer, build_transformer_encoder, build_cross_self_encoder_fixcross
from ms_cm.position_encoding import build_position_encoding
from ms_cm.vslnet_model.layers import ConditionedPredictor, CQConcatenate, NPM



def mask_logits(inputs, mask, mask_value=-1e30):
    mask =  mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value
        
def pad_text_tenor_list(sequences, dtype=torch.long, device=torch.device("cuda"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]

    max_length = max(lengths)

    assert "torch" in str(dtype), "dtype and input type does not match"
    padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


class MS_CM(nn.Module):
    """ This is the module for Multi-scale Cross-modal Transformer  """

    def __init__(self, cross_self_encoder, position_embed, txt_position_embed, txt_dim, vid_dim,
                 input_dropout, conditioned_span_pred_dict,
                 video_frame_contrastive_loss=False, contrastive_hdim=64,
                 max_v_l=75, use_txt_pos=False, n_input_proj=2,NPM_score_norm=0,args=None):
        """ Initializes the model.
        Parameters:
            cross_self_encoder: Our self define Cross-modal Transformer. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            video_frame_contrastive_loss: If true, enable the loss
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
        """
        super().__init__()
        self.cross_self_encoder = cross_self_encoder
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = cross_self_encoder.d_model
        self.max_v_l = max_v_l

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.bs =args.bsz
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.video_frame_contrastive_loss = video_frame_contrastive_loss
        if video_frame_contrastive_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.highlight_proj = nn.Linear(hidden_dim, 1)
        self.saliency_proj = nn.Linear(hidden_dim, 1)

        self.NPM_score_norm = NPM_score_norm
        self.NPM_proj = NPM(hidden_dim)

        self.predictor = ConditionedPredictor(
            dim=hidden_dim,
            num_heads=conditioned_span_pred_dict['num_heads'],
            drop_rate=conditioned_span_pred_dict['drop_rate'],
            max_pos_len=conditioned_span_pred_dict['max_pos_len'],
            predictor=conditioned_span_pred_dict['predictor'],
        )

    def forward(self, src_txt, src_txt_mask , src_vid_total, src_vid_mask_total, scale_mask):
        """
            - src_txt: [batch_size, L_txt, D_txt]
            - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer


            - src_vid_total: a list with N length, and the shape of the list item is [batch_size, L_vid/N, D_vid]
                             
            - src_vid_mask_total: a list with N length, and the shape of the list item is [batch_size, L_vid/N], containing 0 on padded pixels,
                                                                                                will convert to 1 as padding later for transformer
            N is current multiscale choice for this iteration.
        """

        src_txt = self.input_txt_proj(src_txt)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        vid_mem_total_list = []
        vid_mem_withNPM_total_list = []
        NPM_split_score_list = []
        
        out = dict()
        for src_vid,src_vid_mask in zip(src_vid_total, src_vid_mask_total):
            src_vid = self.input_vid_proj(src_vid)
            # TODO should we remove or use different positional embeddings to the src_txt?
            pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
            pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
            # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
            vid_mem_split, txt_mem = self.cross_self_encoder(src_vid, src_vid_mask, src_txt, src_txt_mask, pos_vid, pos_txt)

            NPM_split_score = self.NPM_proj(vid_mem_split,src_vid_mask)
            NPM_split_score_list.append(NPM_split_score)
            vid_mem_total_list.append(vid_mem_split)

        NPM_score_total = torch.cat(NPM_split_score_list,dim=-1)
        if self.NPM_score_norm:

            NPM_score_total_tmp = torch.cat(NPM_split_score_list,dim=-1)
            max_NPMvalue = torch.max(NPM_score_total_tmp,dim=-1)[0]
            NPM_score_total_tmp = NPM_score_total_tmp/max_NPMvalue.unsqueeze(-1)
            NPM_split_score_list = torch.split(NPM_score_total_tmp,1,dim=-1)
        
        for vid_mem_split_tmp , NPM_split_score_tmp in zip(vid_mem_total_list, NPM_split_score_list):
            vid_mem_withNPM_split = vid_mem_split_tmp * NPM_split_score_tmp.unsqueeze(-1)
            vid_mem_withNPM_total_list.append(vid_mem_withNPM_split)


        vid_mem_withNPM_total = torch.cat(vid_mem_withNPM_total_list,dim=1)

        src_vid_mask_final = torch.cat(src_vid_mask_total,dim=1)
        out["saliency_scores"] = self.saliency_proj(vid_mem_withNPM_total).squeeze(-1)  # (bsz, L_vid)

        if self.video_frame_contrastive_loss:
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(src_txt), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem_withNPM_total), p=2, dim=-1)
            out.update(dict(
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))


        cq_vid_text = vid_mem_withNPM_total

        hight_logits = self.highlight_proj(cq_vid_text).squeeze(-1)
        hight_logits = mask_logits(hight_logits, src_vid_mask_final)
        h_score = nn.Sigmoid()(hight_logits)

        highlight_feature = cq_vid_text

        out["NPM_scores"] = NPM_score_total # (bsz, num_scale)
        out["highlight_scores"] = h_score  # (bsz, L_vid)
        out["video_mask"] = src_vid_mask_final
        out["scale_mask"] = scale_mask
        out["txt_mask"] = src_txt_mask

        start_logits, end_logits = self.predictor(highlight_feature, mask=src_vid_mask_final)
        out['start_logits'] = start_logits
        out['end_logits'] =  end_logits

        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for Our method."""

    def __init__(self, weight_dict, losses, temperature, max_v_l,
                 saliency_margin=1):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        # self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

    
    def loss_saliency(self, outputs, targets, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_highlight(self, outputs, targets, log=True):
        """highlight predition"""
        if "highlight_label" not in targets:
            return {"loss_highlight": 0}
        highlight_scores = outputs["highlight_scores"]  # (N, L)
        labels = targets['highlight_label']
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction="none")(highlight_scores, labels)
        loss_per_location = loss_per_location * weights
        mask = outputs['video_mask']
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + 1e-12)
        return {"loss_highlight": loss.mean()}

    def loss_NPM(self, outputs, targets, log=True):
        """NPM predition"""
        if "NPM_labels" not in targets:
            return {"loss_NPM": 0}
        NPM_scores = outputs["NPM_scores"]  # (N, L)
        labels = targets['NPM_labels']

        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction="none")(NPM_scores, labels)
        loss_per_location = loss_per_location * weights
        mask = outputs['scale_mask']
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + 1e-12)
        return {"loss_NPM": loss.mean()}

    def loss_start_end(self, outputs, targets, log=True):
        start_loss = nn.CrossEntropyLoss(reduction="mean")(outputs['start_logits'], targets['start_label'])
        end_loss = nn.CrossEntropyLoss(reduction="mean")(outputs['end_logits'], targets['end_label'])
        return {"loss_start_end": start_loss + end_loss}



    def loss_video_frame_contrastive(self, outputs, targets, log=True):

        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_vid_embed = outputs["proj_vid_mem"]  # (bsz, #queries, d)

        txt_mask = outputs["txt_mask"]
        vid_mask = outputs["video_mask"]

        start_label = targets['start_label']
        end_label = targets['end_label']

        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_vid_embed, normalized_text_embed)  # (bsz, #queries, #tokens)

        logits = logits * txt_mask.unsqueeze(1)
        logits = logits.sum(2) / self.temperature / txt_mask.unsqueeze(1).sum(2)# (bsz, #queries)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)

        for batch_idx, (start_label_tmp,end_label_tmp) in enumerate(zip(start_label,end_label)):
            for idx in range(start_label_tmp,end_label_tmp+1):
                positive_map[batch_idx][idx] = True

        positive_map = positive_map.float()

        neg_mask = 1-positive_map

        mask = positive_map

        neg_logits = torch.exp(logits) * neg_mask * vid_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = -logits + torch.log(exp_logits + neg_logits)

        loss_nce = (mask * log_prob).sum(1) / mask.sum(1)

        losses = {"loss_video_frame_contrastive": loss_nce.mean()}
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            # "spans": self.loss_spans,
            # "labels": self.loss_labels,
            "video_frame_contrastive": self.loss_video_frame_contrastive,
            "highlight": self.loss_highlight,
            "saliency": self.loss_saliency,
            'NPM': self.loss_NPM,
            'start_end': self.loss_start_end
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # print(outputs.keys())
        # exit()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            raise NotImplementedError
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if "saliency" == loss or "highlight" == loss:   # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    device = torch.device(args.device)

    cross_self_encoder = build_cross_self_encoder_fixcross(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    conditioned_span_pred_dict = dict()
    conditioned_span_pred_dict['num_heads']=args.conditioned_span_pred_num_heads
    conditioned_span_pred_dict['drop_rate']=args.conditioned_span_pred_drop_rate
    conditioned_span_pred_dict['max_pos_len']=args.max_v_l
    conditioned_span_pred_dict['predictor']=args.conditioned_span_pred_predictor
    model = MS_CM(
        cross_self_encoder,
        position_embedding,
        txt_position_embedding,
        conditioned_span_pred_dict=conditioned_span_pred_dict,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        video_frame_contrastive_loss=args.video_frame_contrastive_loss,
        contrastive_hdim=args.contrastive_hdim,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        NPM_score_norm=args.NPM_score_norm,
        args=args,
    )

    weight_dict = {
                   "loss_saliency": args.lw_saliency,
                   "loss_highlight": args.lw_highlight,
                   "loss_start_end": args.lw_start_end,
                   "loss_NPM": args.lw_npm,
                   }
    if args.video_frame_contrastive_loss:

        weight_dict["loss_video_frame_contrastive"] = args.video_frame_contrastive_loss_coef

    # losses = ['spans', 'labels', 'highlight', 'saliency']
    losses = ['start_end','highlight', 'saliency','NPM']
    if args.video_frame_contrastive_loss:
        losses += ["video_frame_contrastive"]
    criterion = SetCriterion(
        weight_dict=weight_dict, losses=losses,
        temperature=args.temperature,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin
    )
    criterion.to(device)
    return model, criterion
