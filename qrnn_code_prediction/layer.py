import torch
import torch.nn as nn

'''
Z = tanh(W_z * X)
F = sigmoid(W_f * X)
O = sigmoid(W_o * X)
i = sigmoid(W_i * X)
input : batch_size x embedding_size x time_length
'''

class ConvLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(input_size, output_size*4, kernel_size=2, padding=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)[:, :, :-1] # batch_size x output_size*3 x len(seq)
        z, f, o, i = out.chunk(4, dim=1) # batch x output_size x len(input)-1
        z = self.tanh(z)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        i = self.sigmoid(i)

        return z, f, o, i

class LSTMLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMLayer, self).__init__()

        self.conv = ConvLayer(input_size, output_size)

    def forward(self, x):
        z, f, o, i = self.conv(x)
        c = 0
        time_length = x.size(2)
        for i in range(time_length):
            c = f * c + i * z
            h = o * c

        return h
