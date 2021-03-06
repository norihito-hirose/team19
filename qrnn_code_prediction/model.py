import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from tqdm import tqdm
import pickle
import time

from network import *
from utils import *
from config import cfg

from tensorboard import summary
from tensorboard import FileWriter

class QRNN(object):
    def __init__(self, data):
        if cfg.TRAIN.FLAG:
            self.model_dir = cfg.NET
            self.log_dir = cfg.TRAIN.LOG_DIR
            self.summary_writer = FileWriter(self.log_dir)

        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.gpu = int(cfg.GPU_ID)
            torch.cuda.set_device(self.gpu)
            print("Use CUDA")

        self.epoch = cfg.TRAIN.NUM_EPOCH
        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.vocab_size = data.vocab_size
        self.embedding_size = cfg.EMBEDDING_SIZE
        self.index2word = data.index2word
        self.seqs = data.indexed_seqs

        self.net = self.load_network()

        self.data = data

    def load_network(self):
        net = Net(self.vocab_size)
        if self.cuda:
            net = net.cuda()
        if not cfg.TRAIN.FLAG:
            state_dict = torch.load(cfg.NET, map_location=lambda storage, loc: storage)
            print("Load from", cfg.NET)

        return net

    def train(self):
        start_idx = 0

        net = self.net
        criterion = nn.CrossEntropyLoss()
        lr = cfg.TRAIN.LEARNING_RATE
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.99))

        count = 0
        start_idx = 0
        iteration = len(self.seqs) // self.batch_size + 1
        for epoch in range(self.epoch):
            for i in tqdm(range(iteration)):
                x, lengths = prepare_batch(self.seqs, start_idx, self.batch_size)
                if self.cuda:
                    x = Variable(torch.LongTensor(x)).cuda()
                else:
                    x = Variable(torch.LongTensor(x))

                logits = net(x)
                loss = []

                for i in range(logits.size(0)):
                    logit = logits[i][:lengths[i]-1]
                    target = x[i][1:lengths[i]]
                    loss.append(criterion(logit, target))
                loss = sum(loss) / len(loss)
                net.zero_grad()
                loss.backward()
                optimizer.step()

            end = time.time()

            summary_net = summary.scalar("Loss", loss.data[0])
            self.summary_writer.add_summary(summary_net, count)
            count += 1
            print("epoch %d done. train loss: %f" % (epoch + 1, loss.data[0]))

            if count % cfg.TRAIN.LR_DECAY_INTERVAL == 0:
                lr = lr * 0.95
                optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
                print("decayed learning rate")


        save_model(net, self.model_dir)
        f = open("output/Data/data.pkl", "wb")
        pickle.dump(self.data, f)
        f.close

    def predict(self, seq):
        pass
