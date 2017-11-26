import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import argparse, time

from prepare_data import prepare_sequence
from lstm_model import LSTMModel

def train_model(train_data, train_target, word_to_idx, target_to_idx, model_file = "model.pth", 
    model_type = "LSTM", embedding_dim = 32, hidden_dim = 16, epochs = 10, learning_rate = 0.1, seed = 19):

    torch.manual_seed(seed)

    ## initialize model
    if model_type == "LSTM":
        model = LSTMModel(embedding_dim, hidden_dim, len(word_to_idx), len(target_to_idx))

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    st = time.time()

    print("training model ...")

    # reference: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    count = 0
    loss_mean = 0
    for epoch in range(epochs): 
        for sentence, next_word in zip(train_data, train_target):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = prepare_sequence(sentence, word_to_idx)
            targets = prepare_sequence(next_word, target_to_idx)

            # Step 3. Run our forward pass.
            scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores, targets)
            loss.backward()
            optimizer.step()
            
            loss_mean += loss.data[0]
            
            if count % 100 == 0 and count > 0:
                print("%d sentence done. loss mean: %f" % (count,loss_mean/100))
                loss_mean = 0
            
            count += 1
            
        print("%d th epoch done. %f sec" % (epoch, time.time() - st))

    return model
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training RNN Language Model for Code Prediction')
    parser.add_argument('--datafiles', type=str, default="../data/input/*.txt",
                        help='input file pattern')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='model file')
    parser.add_argument('--word_to_idx', type=str, default='word_to_idx.npy',
                        help='dictionary file (input word -> idx)')
    parser.add_argument('--target_to_idx', type=str, default='target_to_idx.npy',
                        help='dictionary file (target word -> idx)')

    parser.add_argument('--type', type=str, default='LSTM',
                        help='type of model')    
    parser.add_argument('--emdim', type=int, default=32,
                        help='size of word embeddings')
    parser.add_argument('--hiddim', type=int, default=16,
                        help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    parser.add_argument('--seed', type=int, default=19,
                        help='random seed')
 
    #for debug    
    parser.add_argument('--datanum', type=int, default=-1,
                        help='training only first N samples. (for debug)')
    
    args = parser.parse_args()    
    
    ## example of training from files & output model files

    from prepare_data import get_sentences_from_files, prepare_train_data
    
    sentences = get_sentences_from_files(args.datafiles)
    train_data, train_target, word_to_idx, target_to_idx = prepare_train_data(sentences)

    np.save(args.word_to_idx, word_to_idx)
    np.save(args.target_to_idx, target_to_idx)

    if args.datanum > 0:
        train_data = train_data[:args.datanum]
        train_target = train_target[:args.datanum]

    trained_model = train_model(train_data, train_target, word_to_idx, target_to_idx,
                                model_type = args.type, embedding_dim = args.emdim, hidden_dim = args.hiddim, epochs = args.epochs, learning_rate = args.lr, seed = args.seed)

    torch.save(trained_model, args.model)
