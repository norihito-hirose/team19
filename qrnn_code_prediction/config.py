import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.GPU_ID = "0"
__C.WORKERS = 6

__C.NET = "output/Model"
__C.EMBEDDING_SIZE = 200

__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_EPOCH = 2
__C.TRAIN.LEARNING_RATE = 1e-04
__C.TRAIN.LOG_DIR = "output/Log"
