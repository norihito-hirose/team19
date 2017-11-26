RNN/LSTM によるコード予測を行うための python モジュール

# 環境

* Python 3.5.2
* Pytorch 0.2.0_4
* numpy 1.13.3

# ファイル構成

* train.py  モデルの学習、出力を行う
* predict.py  モデルのロード、次のコードを予測する
* prepare_data.py  学習・予測用データの準備のためのモジュール
* lstm_model.py  LSTMモデルクラス(LSTMModel)の定義

それぞれの関数の使い方は train.py または predict.py の main にあるコードを参照してください。

# コマンドラインでの動かし方

## 学習

ヘルプ

`python train.py -h`

例: デバッグ用に、初めの200データのみ、1 epochのみで学習

`python train.py --epochs 1 --datanum 200`

## 予測

ヘルプ

`python main predict.py -h`

例: ソースファイルを指定して、次のコードを予測、上位10件をプリント。学習済みの model.pth, *_to_idx.npy のパスを指定。

`python predict.py predict_sample/xor_tensorflow.py --model ../model/model_lstm.pth --word_to_idx ../model/word_to_idx.npy --target_to_idx ../model/target_to_idx.npy --topN 10`
