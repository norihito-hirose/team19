import torch
from torch.autograd import Variable
import argparse
import pickle

from utils import *
from model import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Code Prediction using RNN Language Model')
    parser.add_argument("file")
    args = parser.parse_args()

    import re
    def find_tf(file_path):
        f = open(file_path, "r")
        code = f.read()
        elements = re.findall("tf\.[\w|\.]*", code)
        return elements

    sentence = find_tf(args.file)
    print(sentence)

    f = open("output/Data/data.pkl", "rb")
    data = pickle.load(f)
    sentence = data.index_from_seq(sentence)
    sentence = Variable(torch.LongTensor(sentence))

    model = QRNN(data)
    result = QRNN.predict_topN(sentence)
    print(sentence)
