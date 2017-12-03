import unittest
from lstm_model import LSTMModel

class LSTMModelTestCase(unittest.TestCase):
    """ test lstm model """
    def test_init(self):
        self.assertIsInstance(LSTMModel(1, 1, 1, 1), LSTMModel)

if __name__ == '__main__':
    unittest.main()
