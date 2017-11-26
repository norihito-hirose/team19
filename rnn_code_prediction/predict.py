import torch
import numpy as np
import argparse

from prepare_data import START_OF_SENTENCE, END_OF_SENTENCE, prepare_sequence
  
def load_model(model_file = "model.pth", word_to_idx_file = "word_to_idx.npy", target_to_idx_file = "target_to_idx.npy"):
    # load trained model, word_to_idx, target_to_idx
    model = torch.load(model_file)
    word_to_idx = np.load(word_to_idx_file).item()
    target_to_idx = np.load(target_to_idx_file).item()

    target_word_list = [None for i in range(len(target_to_idx))]
    for word, idx in target_to_idx.items():
        target_word_list[idx] = word

    return (model, word_to_idx, target_to_idx, target_word_list)

def predict_topN(target_word_list, score, N, with_probability = False, eliminate_eos = False):
    if eliminate_eos:
        tmp = np.argsort(score)[::-1]
        candIdx = []
        for i in tmp:
            if target_word_list[i] != END_OF_SENTENCE:
                candIdx.append(i)
            if len(candIdx) == N:
                break
    else:
        candIdx = np.argsort(score)[::-1][:N]

    if with_probability:
        # log softmax -> softmax (probability)
        return [(target_word_list[i], np.exp(score[i])) for i in candIdx]
    else:
        return [target_word_list[i] for i in candIdx]

def predict_next_code(sentence, model, word_to_idx, target_word_list, topN = 5):
    if sentence[0] != START_OF_SENTENCE:
        sentence = [START_OF_SENTENCE] + sentence 
    inputs = prepare_sequence(sentence, word_to_idx)
    scores = model(inputs)
    return predict_topN(target_word_list, scores.data.numpy()[-1], topN, 
                with_probability = True, eliminate_eos = True)

if __name__ == '__main__':

    # example of predicting next code from code file
    
    parser = argparse.ArgumentParser(description='Code Prediction using RNN Language Model')
    parser.add_argument("file")
    parser.add_argument('--model', type=str, default='model.pth',
                        help='model file')
    parser.add_argument('--word_to_idx', type=str, default='word_to_idx.npy',
                        help='dictionary file (input word -> idx)')
    parser.add_argument('--target_to_idx', type=str, default='target_to_idx.npy',
                        help='dictionary file (target word -> idx)')
    parser.add_argument('--topN', type=int, default=5,
                        help='top N prediction')
    args = parser.parse_args()
    
    # from ../preprocess.py
    # extracet tf.hoge from code
    import re
    def find_tf(file_path):
        f = open(file_path, "r")
        code = f.read()
        elements = re.findall("tf\.[\w|\.]*", code)
        return elements
    
    sentence = find_tf(args.file)    
    print(sentence)

    model, word_to_idx, __, target_word_list = load_model(args.model, args.word_to_idx, args.target_to_idx)
    print(predict_next_code(sentence, model, word_to_idx, target_word_list, args.topN))
