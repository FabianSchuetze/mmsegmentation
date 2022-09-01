import sys
import os.path as osp
import mmcv
import copy
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import random
# convert dataset annotation to semantic segmentation map
data_root =sys.argv[1]
img_dir = 'images'
ann_dir = 'labels'
classes = ('bridge',)
palette = [[128, 128, 128]]
# define class and plaette for better visualization

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
random.shuffle(filename_list)
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*99/100)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line + '\n' for line in filename_list[train_length:])



#@DATASETS.register_module()
class Bridge(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None

from mmcv import Config
cfg =\
Config.fromfile('configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py')
# Config.fromfile('configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py')
# Config.fromfile('configs/sem_fpn/fpn_r101_512x512_160k_ade20k.py')

from mmseg.apis import set_random_seed

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.auxiliary_head.norm_cfg = dict(type='BN', requires_grad=True)
cfg.device = 'cuda'
cfg.model.decode_head.num_classes = 2
# cfg.model.decode_head.loss_decode = \
    # dict(type='MyLoss', loss_weight=1.0)


# Modify dataset type and path
cfg.dataset_type = 'Bridge'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu=4

cfg.img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)

img_scale=(720, 960)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MyAugmentations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(1024, 960), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(960, 720)),
    dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=img_scale, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
    dict(
       type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    # dict(type='LoadAnnotations'),
]
cfg.model.test_cfg.mode = 'whole'


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/tutorial'

cfg.runner.max_iters = 20000
cfg.log_config.interval = 400
cfg.log_config.hooks.append(
        dict(type='MMSegWandbHook',
         init_kwargs={'project': 'MMDetection-tutorial'},
         log_checkpoint=False,
         num_eval_images=1,
         interval=400))
# cfg.evaluation.hooks.append(
        # dict(type='EvalHook'))
         # init_kwargs={'project': 'MMDetection-tutorial'},
         # efficient_test=True,
         # log_checkpoint=True,
         # num_eval_images=1,
         # interval=10))
cfg.evaluation.interval = 400
# cfg.evaluation.metric = None
cfg.evaluation.pre_eval = False
cfg.evaluation.out_dir = '/tmp/'
# cfg.evaluation.num_eval_images = 2
cfg.checkpoint_config.interval = 500
cfg.checkpoint_config.meta = dict(
        CLASSES=classes,
        PALETTE=palette)
cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.data.samples_per_gpu / 8
# optimizer = dict(type='Adam', lr=0.0002)

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training



# Build the dataset
# datasets = [build_dataset(cfg.data.train)]
datasets = [build_dataset(cfg.data.train)]

# Build the detector
# Add an attribute for visualization convenience
cfg.workflow = [('train', 1), ('val', 1)]
if len(cfg.workflow) == 2:
  val_dataset = copy.deepcopy(cfg.data.val)
  val_dataset.pipeline = cfg.data.train.pipeline
  datasets.append(build_dataset(val_dataset))
# Create work_dir
model = build_segmentor(cfg.model)
model.CLASSES = datasets[0].CLASSES
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())

