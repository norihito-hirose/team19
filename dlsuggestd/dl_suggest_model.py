""" This code is predict model """
import logging
from predict import load_model, predict_next_code

class DLSuggestModel(object):
    """
    predict next code
    """

    def __init__(self, torch_model_file, word_idx_file, target_idx_file):
        self.info = {
            'model_file': torch_model_file,
            'word_to_idx_file': word_idx_file,
            'target_to_idx_file': target_idx_file
        }
        print("Loading trained model: %s" % torch_model_file)
        print("Loading word idx: %s" % word_idx_file)
        print("Loading target idx: %s" % target_idx_file)
        self.model, self.word_to_idx, __, self.target_word_list = load_model(
            torch_model_file, word_idx_file, target_idx_file)

    def predict(self, sentence, top_n=1):
        """ predict next sentence """
        predicts = predict_next_code(
            sentence, self.model, self.word_to_idx, self.target_word_list, top_n)
        return {
            'info': self.info,
            'candidates': self.render_dict(predicts)
        }

    def render_dict(self, predicts): # pylint: disable=no-self-use
        """ render dict """
        candidates = []
        for pred in predicts:
            logging.debug(type(pred[1]))

            candidates.append({'code': pred[0], 'probability': pred[1].item()})
        return candidates

    def get_info(self, path):
        self.info['request_url'] = path
        return self.info
