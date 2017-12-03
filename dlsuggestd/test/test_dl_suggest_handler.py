""" This script is testing DLSuggestHandler """
import unittest
import os
from dl_suggest_model import DLSuggestModel
from dl_suggest_handler import DLSuggestHandler

# pylint: disable=invalid-name
model_file = os.path.abspath('../model/model_lstm.pth')
word_idx_file = os.path.abspath('../model/word_to_idx.npy')
target_idx_file = os.path.abspath('../model/target_to_idx.npy')

def _create_model():
    model = DLSuggestModel(model_file, word_idx_file, target_idx_file)
    DLSuggestHandler.model = model

class DLSuggestHandlerTestCase(unittest.TestCase):
    """ test DLSuggestHandler """

if __name__ == '__main__':
    unittest.main()
