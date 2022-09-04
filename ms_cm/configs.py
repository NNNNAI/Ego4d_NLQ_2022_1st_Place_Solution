import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=400, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # Data config
        parser.add_argument("--max_q_l", type=int, default=32)
        parser.add_argument("--max_v_l", type=int, default=75)
        parser.add_argument("--clip_length", type=int, default=2)
        parser.add_argument("--max_windows", type=int, default=5)

        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")

        # Model config
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--onlyv_layers', default=2, type=int,
                            help="Number of onlyv layers in the transformer")
        parser.add_argument('--enc_layers', default=2, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=2, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')
        # other model configs
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
        parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce video_frame_contrastive_loss")
        
        # Loss weight
        parser.add_argument("--lw_saliency", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_highlight", type=float, default=5.,
                            help="weight for highlight loss, set to 0 will ignore")
        parser.add_argument("--lw_npm", type=float, default=1.,
                            help="weight for NPM loss, set to 0 will ignore")
        parser.add_argument("--lw_start_end", type=float, default=1,
                            help="weight for start end ce loss, set to 0 will ignore")

        parser.add_argument("--video_frame_contrastive_loss_coef", default=0.0, type=float,\
                            help="weight for video frame-level contrastive loss")


        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--video_frame_contrastive_loss", action="store_true",
                            help="Disable video_frame_contrastive_loss.")

        # Param for data pre-process
        parser.add_argument("--vsldataset_task", type=str, default="charades", help="vsldataset target task")
        parser.add_argument(
            "--vsldataset_fv", type=str, default="new", help="[new | org] for visual features"
        )
        parser.add_argument(
            "--vsldataset_num_workers",
            type=int,
            default=32,
            help="Number of CPU workers to process the data",
        )
        parser.add_argument("--vslnet_datapath", type=str, default="vslnet_datapath", help="vsldataset path")
        parser.add_argument("--vslnet_thres_in_train", type=int, default=-1, help="exclude the windows less than vslnet_thres_in_train size")
        
        parser.add_argument(
            "--vslnet_dataset_save_dir",
            type=str,
            default="vslnetdatasets",
            help="path to save processed dataset",
        )
        parser.add_argument(
            "--eval_gt_json",
            type=str,
            default=None,
            help="Provide GT JSON to evaluate while training"
        )
        parser.add_argument("--conditioned_span_pred_num_heads", type=int, default=8, help="number of heads")
        parser.add_argument("--conditioned_span_pred_drop_rate", type=float, default=0.2, help="dropout rate")
        parser.add_argument("--conditioned_span_pred_predictor", type=str, default="bert", choices=['rnn', 'bert','justfc'])

        # param for cross modal encoder
        parser.add_argument('--num_cross_encoder_layers', default=2, type=int,
                            help="Number of encoding layers in the cross_encoder")
        parser.add_argument("--cross_text_use_ori", action="store_true",
                            help="whether to use ori clip text feature in self first method, defalt is using the text feature from transformer encoder")
        parser.add_argument('--cross_input_dropout', default=0.1, type=float,
                            help="Dropout applied in the cross encoder if in cross first mode")
        parser.add_argument("--cross_first", action="store_true",
                            help="whether to use cross first mode, defalut is cross encoder first and then self encoder")


        parser.add_argument('--v_hidden_size', default=256, type=int,
                            help="Size of the embeddings in video size in cross encoder")

        parser.add_argument('--hidden_size', default=256, type=int,
                            help="Size of the embeddings in text size in cross encoder")
        parser.add_argument(
            "--v_hidden_act", type=str, default="gelu", help="activation in cross encoder for video side"
        )
        parser.add_argument(
            "--hidden_act", type=str, default="gelu", help="activation in cross encoder for text side"
        )

        parser.add_argument('--v_intermediate_size', default=1024, type=int,
                            help="intermediate_size in the cross output layer for video side")

        parser.add_argument('--intermediate_size', default=1024, type=int,
                            help="intermediate_size in the cross output layer for text side")\
        
        parser.add_argument("--v_hidden_dropout_prob", type=float, default=0.1, help="dropout rate for cross encoder in video side")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="dropout rate for cross encoder in text side")

        parser.add_argument('--bi_num_attention_heads', default=8, type=int,
                            help="bi_num_attention_heads")

        parser.add_argument('--bi_hidden_size', default=256, type=int,
                            help="bi_hidden_size")

        parser.add_argument('--v_target_size', default=2818, type=int,
                            help="v_target_size")


        parser.add_argument("--v_attention_probs_dropout_prob", type=float, default=0.1, help="dropout rate for cross encoder attention in video side")
        parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="dropout rate for cross encoder attention in text side")

        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for BertAdam. "
                "E.g., 0.1 = 10%% of training.",
        )
        parser.add_argument("--ego4d_train_num", type=int, default=11296, help="number of ego4d train dataset")

        parser.add_argument("--optim_name", type=str, default= 'AdamW', choices=['AdamW', 'BertAdam'])

        parser.add_argument("--numscale_list", type=int, nargs="+",
                            help="num_scale_list "
                                 "scale list for multiscale strategy")

        parser.add_argument("--allscaletopk", type=int,default=2,
                            help="allscaletopk")
        parser.add_argument("--NPM_score_norm", default=0, type=int,
                            help="whether norm the NPM score in the mul stage")
                            
        parser.add_argument("--use_sw", default=0, type=int,
                            help="use Variable-length Sliding Window Sampling (SW) as data agrumentation")

        parser.add_argument("--sw_len_ratio", type=float, nargs="+",
                            help="length ratio interval used for SW, like [0.5,0.8]")

        parser.add_argument("--use_vs", default=0, type=int,
                            help="use Video Splicing (VS) as data agrumentation as data agrumentation")

        parser.add_argument('--vs_prob', default=0.5, type=float,
                            help="Probablity of using VS for each data sample")

        parser.add_argument("--preds_list", type=str, nargs="+",
                            help="preds_list "
                                 "preds checkpoint list")
        
        
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if isinstance(self, TestOptions):
            opt.model_dir = os.path.dirname(opt.resume)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers",
                               "resume", "resume_all"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([ctx_str, opt.exp_id]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            # code_dir = os.path.dirname(os.path.realpath(__file__))
            # code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            # make_zipfile(code_dir, code_zip_filename,
            #              enclosing_dir="code",
            #              exclude_dirs_substring="results",
            #              exclude_dirs=["results", "debug_results", "__pycache__"],
            #              exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--split", type=str,
                                 help="use_whichsplit for eval")
