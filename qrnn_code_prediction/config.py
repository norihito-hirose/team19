import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.GPU_ID = "0"
__C.WORKERS = 6

__C.NET = "output/Model"
__C.EMBEDDING_SIZE = 200

__C.DATA = edict()
__C.DATA.MIN_FREQUENCY = 800
__C.DATA.MIN_LENGTH = 3
__C.DATA.MAX_LENGTH = 400

__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_EPOCH = 150
__C.TRAIN.LR_DECAY_INTERVAL = 1000
__C.TRAIN.LEARNING_RATE = 1e-04
__C.TRAIN.LOG_DIR = "output/Log"
