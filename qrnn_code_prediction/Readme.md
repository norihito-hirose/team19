QRNN によるコード予測を行うための python モジュール

# 環境

* Python 3.6.3
* Pytorch 0.4.0a0+0a434ff
* tensorboard 1.0.0a6
* tensorboardX 0.8

# ファイル構成

* network.py QRNNのネットワークと層の定義
* model.py trainとpredict(WIP)の関数の定義
* train.py モデルの学習
* utils.py 各種補助の関数とクラスの定義
* cofig.py trainのためのconfig記述ファイル

# コマンドラインでの動かし方
## 学習
`python train.py`
