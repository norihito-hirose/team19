# reference: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to target space
        self.decoder = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        target_space = self.decoder(lstm_out.view(len(sentence), -1))
        scores = F.log_softmax(target_space)
        return scores
    