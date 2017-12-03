""" This script is testing DLSuggestModel """
import unittest
import os
import numpy as np
from dl_suggest_model import DLSuggestModel

# pylint: disable=invalid-name
model_file = os.path.abspath('../model/model_lstm.pth')
word_idx_file = os.path.abspath('../model/word_to_idx.npy')
target_idx_file = os.path.abspath('../model/target_to_idx.npy')

def _create_model():
    return DLSuggestModel(model_file, word_idx_file, target_idx_file)

class DLSuggestModelTestCase(unittest.TestCase):
    """ test DLSuggestModel """
    def test_render_dict(self):
        """ test render dict """
        model = _create_model()
        input_sentence = [('tf.ones', np.float32(0.123)), ('tf.zeros', np.float32(0.023))]
        result = model.render_dict(input_sentence)
        self.assertEqual(len(result), 2)
        self.assertTrue('code' in result[0])
        self.assertTrue('probability' in result[0])

    def test_predict(self):
        """ test predict """
        model = _create_model()
        top_n = 2
        result = model.predict(['tf.ones'], top_n)
        expect_info = {
            'model_file': model_file,
            'target_to_idx_file': target_idx_file,
            'word_to_idx_file': word_idx_file
        }

        self.assertTrue('candidates' in result)
        self.assertEqual(len(result['candidates']), top_n)
        self.assertTrue('info' in result)
        self.assertDictEqual(result['info'], expect_info)

if __name__ == '__main__':
    unittest.main()
