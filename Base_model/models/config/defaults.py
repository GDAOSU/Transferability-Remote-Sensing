from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# experiment name. will be used in the path names \for log- and savefiles
_C.experiment_name = "experiment"
_C.log_dir = "logs"
_C.DIR = "ckpt/experiment"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.num_class = 4
# selected bands
_C.DATASET.selected_bands = [0, 1, 2, 3]
_C.DATASET.use_height = False
_C.DATASET.patch_size = ([512, 512])
# randomly horizontally flip images when train/test
#_C.DATASET.random_flip = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.model = "unet"
_C.MODEL.backbone = "resnet50"
# weights to finetune
_C.MODEL.weights = ""

# deeplab-specific
# initialize ResNet-101 backbone with ImageNet pre-trained weights
_C.MODEL.pretrained_backbone = False
# network output stride
_C.MODEL.out_stride = 16
# Pretrain model for HRNet
_C.MODEL.PRETRAINED = ''
# HRNet setting
_C.MODEL.EXTRA = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()


# training state will be saved every save_freq batches/optimization steps during training
_C.TRAIN.save_freq = 1000
# tensorboard logs will be written every log_freq number of batches/optimization steps
_C.TRAIN.log_freq = 100
# frequency to display
_C.TRAIN.disp_freq = 20

# training hyperparameters
_C.TRAIN.optim = "SGD"
_C.TRAIN.lr = 0.02
_C.TRAIN.lr_pow = 0.9
_C.TRAIN.momentum = 0.9
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.batch_size = 2
_C.TRAIN.workers = 16
_C.TRAIN.start_epoch = 0
_C.TRAIN.num_epoch = 20

# manual seed
_C.TRAIN.seed = 1860

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# validation will be run every val_freq batches/optimization steps during training
_C.VAL.val_freq = 1000
# currently only supports 1
_C.VAL.batch_size = 1

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
# overlapping ratio
_C.TEST.overlap_ratio = 0.0
# overlapping ratio
_C.TEST.score = True
#    London Argentina Haiti OSU Campus Singapore Potsdam DeepGlobe OMA DSTL
_C.TEST.dataset = 'OMA'
_C.TEST.workers = 4
_C.TEST.viz = True