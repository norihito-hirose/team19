from model import QRNN
from utils import *

if __name__ == '__main__':
    data = DataPreprocessor()
    qrnn = QRNN(data)
    qrnn.train()
