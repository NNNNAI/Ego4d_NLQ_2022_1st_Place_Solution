ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results


######## setup video+text features
feat_root=features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(../episodic-memory/NLQ/VSLNet/data/features/nlq_official_v1/official)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(./CLIP_feature_slowfast/Slowfast_vitb16CLIP_videofeature)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=./CLIP_feature_slowfast/Slowfast_vitb16CLIP_textfeature
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# multi_scale param
scale_list=()
scale_list+=(2)
scale_list+=(3)
scale_list+=(4)
scale_list+=(5)
scale_list+=(6)


#### training
bsz=32

sw_len_ratio=()
sw_len_ratio+=(0.4)
sw_len_ratio+=(0.8)

PYTHONPATH=$PYTHONPATH:. python ms_cm/train_ego4d_slowfast.py \
--ctx_mode ${ctx_mode} \
--v_feat_dirs ${v_feat_dirs[@]} \
--numscale_list ${scale_list[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--vsldataset_task nlq_official_v1 \
--vsldataset_fv official \
--vsldataset_num_workers 32 \
--vslnet_datapath ../episodic-memory/NLQ/VSLNet/data/ \
--vslnet_dataset_save_dir ./vslnet_dataset_savepath \
--eval_gt_json ../episodic-memory/NLQ/VSLNet/data/nlq_val.json \
--no_aux_loss \
--cross_first \
--max_v_l $1 \
--lw_saliency 1 \
--lw_highlight 20 \
--enc_layers 0 \
--hidden_dim $2 \
--v_hidden_size $2 \
--bi_hidden_size $2 \
--hidden_size $2 \
--num_cross_encoder_layers $3 \
--dropout $4 \
--v_hidden_dropout_prob $4 \
--hidden_dropout_prob $4 \
--v_attention_probs_dropout_prob $4 \
--attention_probs_dropout_prob $4 \
--vslnet_thres_in_train 0 \
--use_sw ${5} \
--sw_len_ratio ${sw_len_ratio[@]} \
--use_vs ${6} \
--vs_prob ${7} \
--video_frame_contrastive_loss \
--video_frame_contrastive_loss_coef 1 \
--contrastive_hdim ${8} \
--exp_id ${9} \

