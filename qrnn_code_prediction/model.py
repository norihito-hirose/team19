import torch
import torch.nn as nn
from layer import ConvLayer, LSTMLayer

class Net(nn.Module):
    def __init__(self, vocab_size, embedding_size=200):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.Sequential(
            LSTMLayer(embedding_size, 256),
            LSTMLayer(256, 256),
            LSTMLayer(256, 256)
        )
        self.fc = nn.Linear(256, vocab_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.embedding(x)
        out = out.transpose(1, 2)
        out = self.lstm(out)
        out = out.transpose(1, 2)
        out = self.fc(out)
        out = self.log_softmax(out)

        return out
