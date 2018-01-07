import torch
import numpy as np
import glob
from config import cfg

class DataPreprocessor(object):
    def __init__(self, data_dir="../data/input/*.txt"):
        self.data_dir = data_dir
        self.seqs = self.prepare_seqs()
        self.indexed_seqs = []
        self.vocab_size = 0
        self.word2index = {}
        self.index2word = {0: "PAD"}
        self.word2count = {}
        self.trimmed = False
        self.keep_words = []

    def prepare_seqs(self):
        files = glob.glob(self.data_dir)
        seqs = []
        for f in files:
            seq = open(f).read().strip().split("\n")
            if cfg.DATA.MAX_LENGTH >= len(seq) and len(seq) >= cfg.DATA.MIN_LENGTH:
                seqs.append(seq)
        seqs.sort(key=len)

        return seqs

    def index_words(self):
        n_words = 1
        for sentence in self.seqs:
            for word in sentence:
                if word not in self.word2index:
                    self.word2index[word] = n_words
                    self.word2count[word] = 1
                    self.index2word[n_words] = word
                    n_words += 1
                else:
                    self.word2count[word] += 1

        self.vocab_size = n_words

    def indexes_from_seqs(self):
        indexed_seqs = []
        for seq in self.seqs:
            indexed_seq = [self.word2index[word] for word in seq if word in self.keep_words]
            if len(indexed_seq) >= cfg.DATA.MIN_LENGTH:
                indexed_seqs.append(indexed_seq)

        indexed_seqs.sort(key=len)
        self.indexed_seqs = indexed_seqs

    def trim(self, min_frequency=cfg.DATA.MIN_FREQUENCY):
        if self.trimmed:
            return
        else:
            self.trimmed = True

        for word, count in self.word2count.items():
            if count >= min_frequency:
                self.keep_words.append(word)

        self.word2index = {}
        self.index2word = {0: "PAD"}
        self.word2count = {}
        n_words = 1

        for sentence in self.seqs:
            for word in sentence:
                if word in self.keep_words:
                    if word not in self.word2index:
                        self.word2index[word] = n_words
                        self.word2count[word] = 1
                        self.index2word[n_words] = word
                        n_words += 1
                    else:
                        self.word2count[word] += 1

        self.vocab_size = n_words



def prepare_batch(seqs, start_idx, batch_size):
    end_idx = start_idx + batch_size
    if end_idx > len(seqs):
        batched_seq = seqs[start_idx:]
    else:
        batched_seq = seqs[start_idx:start_idx + batch_size]
    max_length = len(batched_seq[-1])
    lengths = batched_seq.copy()
    lengths = list(map(len, lengths))

    for seq in batched_seq:
        seq += [0 for i in range(max_length - len(seq))]

    return batched_seq, lengths

def save_model(Net, model_dir):
    torch.save(Net.state_dict(), "%s/Net.pth" % (model_dir))
    print("Save Model")
