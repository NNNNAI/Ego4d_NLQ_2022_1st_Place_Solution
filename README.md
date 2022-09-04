# Ego4d_NLQ_2022_1st_Place_Solution
The 1st place solution of 2022 Ego4d Natural Language Queries. 
For more details, please refer to our paper: [ReLER@ ZJU-Alibaba Submission to the Ego4D Natural Language Queries Challenge 2022](https://arxiv.org/abs/2207.00383)

# Data preparation
## Ego4d NLQ data 
Pleasr follow the preparation section of [Ego4d's VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet#preparation) to preprocess the raw Ego4d NLQ data.
## Pre-Extracted feature
We provide two Pre-Extracted feature for video and text by using [CLIP](https://github.com/openai/CLIP) (Vit-B/16).
- **Please download the [Slowfast_vitb16CLIP_textfeature.zip](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution/releases/download/data/Slowfast_vitb16CLIP_textfeature.zip) and [Slowfast_vitb16CLIP_videofeature.zip](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution/releases/download/data/Slowfast_vitb16CLIP_videofeature.zip).**
- **Upzip the Slowfast_vitb16CLIP_textfeature.zip and Slowfast_vitb16CLIP_videofeature.zip, place these folders in ./CLIP_feature_slowfast.**

## Quick Start

**Train** and **Test**

```shell
# To train the model.
bash ms_cm/scripts/train_ego4d_slowfast.sh \
     600 256 3 0.1 1 1 0.5 64 vlen600_slowfast
```

```shell
# To predict on eval set.
python -m ms_cm.inference_ego4d_slowfast \
      --resume ./results/video_tef-vlen600_slowfast/model_0146.ckpt --split val
      
# To predict on test set.
python -m ms_cm.inference_ego4d_slowfast \
      --resume ./results/video_tef-vlen600_slowfast/model_0146.ckpt --split test
```
Notably, the resume path "./results/video_tef-vlen600_slowfast/model_0146.ckpt "is just a example, you can change this path with the path of your own trained model checkpoint. We also prvoided the checkpoint: ["model_0146.ckpt"](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution/releases/download/data/video_tef-vlen600_slowfast.zip). If you want to use this chekpoint, just download and upzip it into ./results. 

# Citation
```
@article{liu2022reler,
  title={ReLER@ ZJU-Alibaba Submission to the Ego4D Natural Language Queries Challenge 2022},
  author={Liu, Naiyuan and Wang, Xiaohan and Li, Xiaobo and Yang, Yi and Zhuang, Yueting},
  journal={arXiv preprint arXiv:2207.00383},
  year={2022}
}
```
```
@article{Ego4D2021,
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and Martin, Miguel and Nagarajan, Tushar and Radosavovic, Ilija and Ramakrishnan, Santhosh Kumar and Ryan, Fiona and Sharma, Jayant and Wray, Michael and Xu, Mengmeng and Xu, Eric Zhongcong and Zhao, Chen and Bansal, Siddhant and Batra, Dhruv and Cartillier, Vincent and Crane, Sean and Do, Tien and Doulaty, Morrie and Erapalli, Akshay and Feichtenhofer, Christoph and Fragomeni, Adriano and Fu, Qichen and Fuegen, Christian and Gebreselasie, Abrham and Gonzalez, Cristina and Hillis, James and Huang, Xuhua and Huang, Yifei and Jia, Wenqi and Khoo, Weslie and Kolar, Jachym and Kottur, Satwik and Kumar, Anurag and Landini, Federico and Li, Chao and Li, Yanghao and Li, Zhenqiang and Mangalam, Karttikeya and Modhugu, Raghava and Munro, Jonathan and Murrell, Tullie and Nishiyasu, Takumi and Price, Will and Puentes, Paola Ruiz and Ramazanova, Merey and Sari, Leda and Somasundaram, Kiran and Southerland, Audrey and Sugano, Yusuke and Tao, Ruijie and Vo, Minh and Wang, Yuchen and Wu, Xindi and Yagi, Takuma and Zhu, Yunyi and Arbelaez, Pablo and Crandall, David and Damen, Dima and Farinella, Giovanni Maria and Ghanem, Bernard and Ithapu, Vamsi Krishna and Jawahar, C. V. and Joo, Hanbyul and Kitani, Kris and Li, Haizhou and Newcombe, Richard and Oliva, Aude and Park, Hyun Soo and Rehg, James M. and Sato, Yoichi and Shi, Jianbo and Shou, Mike Zheng and Torralba, Antonio and Torresani, Lorenzo and Yan, Mingfei and Malik, Jitendra},
  title     = {Ego4D: Around the {W}orld in 3,000 {H}ours of {E}gocentric {V}ideo},
  journal   = {CoRR},
  volume    = {abs/2110.07058},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07058},
  eprinttype = {arXiv},
  eprint    = {2110.07058}
}
```
