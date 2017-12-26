import torch
import numpy as np
import glob

class DataPreprocessor(object):
    def __init__(self, data_dir="../data/input/*.txt"):
        self.data_dir = data_dir
        self.seqs = self.prepare_seqs()
        self.word2index, self.index2word = self.index_words()
        self.indexed_seqs = self.indexes_from_seqs()
        self.vocab_size = len(self.word2index) + 1

    def prepare_seqs(self):
        files = glob.glob(self.data_dir)
        seqs = []
        for f in files:
            seq = open(f).read().strip().split("\n")
            if len(seq) < 512 and len(seq) > 2:
                seqs.append(seq)
        seqs.sort(key=len)

        return seqs

    def index_words(self, keep_words=None):
        word2index = {}
        word2count = {}
        index2word = {0: "PAD"}
        n_words = 1
        for sentence in self.seqs:
            for word in sentence:
                if word not in word2index:
                    word2index[word] = n_words
                    index2word[n_words] = word
                    n_words += 1

        return word2index, index2word

    def indexes_from_seqs(self):
        indexed_seqs = []
        for seq in self.seqs:
            indexed_seq = [self.word2index[word] for word in seq]
            indexed_seqs.append(indexed_seq)

        return indexed_seqs

    def save_dict(self):
        np.save()

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
