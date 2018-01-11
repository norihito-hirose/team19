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
        Z, F, O, I = out.chunk(4, dim=1) # batch x output_size x len(input)-1
        Z = self.tanh(Z)
        F = self.sigmoid(F)
        O = self.sigmoid(O)
        I = self.sigmoid(I)

        return Z, F, O, I

class LSTMLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMLayer, self).__init__()

        self.conv = ConvLayer(input_size, output_size)

    def forward(self, x):
        Z, F, O, I = self.conv(x)
        H = None
        c = 0
        time_length = x.size(2)
        for t in range(time_length):
            z = Z[:, :, t]
            f = F[:, :, t]
            o = O[:, :, t]
            i = I[:, :, t]
            c = f * c + i * z
            h = o * c

            if H is None:
                H = h.unsqueeze(2)
            else:
                H = torch.cat((H, h.unsqueeze(2)), dim=2)

        return H

class Net(nn.Module):
    def __init__(self, vocab_size, embedding_size=200):
        super(Net, self).__init__()

        if self.cuda:
            self.embedding = nn.Embedding(vocab_size, embedding_size).cuda()
            self.lstm = nn.Sequential(
                LSTMLayer(embedding_size, 256),
                LSTMLayer(256, 256),
                LSTMLayer(256, 256)
            ).cuda()
            self.fc = nn.Linear(256, vocab_size).cuda()
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.lstm = nn.Sequential(
                LSTMLayer(embedding_size, 256),
                LSTMLayer(256, 256),
                LSTMLayer(256, 256)
            )
            self.fc = nn.Linear(256, vocab_size)


    def forward(self, x):
        out = self.embedding(x)
        out = out.transpose(1, 2)
        out = self.lstm(out)
        out = out.transpose(1, 2)
        out = self.fc(out)

        return out
