from model import QRNN
from utils import *

if __name__ == '__main__':
    data = DataPreprocessor()
    data.index_words()
    data.trim()
    data.indexes_from_seqs()
    qrnn = QRNN(data)
    qrnn.train()
