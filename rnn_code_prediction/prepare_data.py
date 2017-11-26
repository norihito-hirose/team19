import torch
import torch.autograd as autograd
import glob

# special token
UNKNOWN_WORD = "<unknown>"
START_OF_SENTENCE = "<SOS>"
END_OF_SENTENCE = "<EOS>"

def get_sentences_from_files(input_file_pattern = "../data/input/*.txt"):
    input_files = glob.glob(input_file_pattern)

    # view one source file as a sentence
    def get_one_sentence(file):
        with open(file, "r", encoding="utf-8") as fr:
            sentence = [line[:-1] for line in fr.readlines()]
        return sentence

    # return sentences
    return [get_one_sentence(input_file) for input_file in input_files]

def prepare_train_data(sentences):
    # train data
    train_data = [[START_OF_SENTENCE] + s for s in sentences]

    # target values (next words)
    train_target = [sentence[1:] + [END_OF_SENTENCE] for sentence in train_data]

    def get_word_to_idx_dict(sentences):
        word_to_idx = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx

    # get word to index dictionary
    word_to_idx = get_word_to_idx_dict(train_data)            
    target_to_idx = get_word_to_idx_dict(train_target)    

    # prepare for unknown word input
    word_to_idx[UNKNOWN_WORD] = len(word_to_idx)

    return (train_data, train_target, word_to_idx, target_to_idx)

def prepare_sequence(seq, idx_dict):
    idxs = [idx_dict[w] if w in idx_dict else idx_dict[UNKNOWN_WORD] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
