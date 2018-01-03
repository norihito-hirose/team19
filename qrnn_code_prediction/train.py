from model import QRNN

if __name__ == '__main__':
    data = DataPreprocessor()
    qrnn = QRNN(data)
    qrnn.train()
