dlsuggestd
==============================================

predict next sentence for program languages tcp daemon server.

# install dependencies

```bash
$ pip3 install -r requirements.txt
```

# Usage

set PYTHONPATH for lstm codes.

```bash
$ export PYTHONPATH=$team19/rnn_code_prediction:$PYTHONPATH
```

replace `$team19` to your location this repository.

## start daemon

```bash
$ bin/dlsuggestd start -h
usage: dlsuggestd start [-h] [-p PORT] -m MODEL -w WORD_IDX -t TARGET_IDX

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  listen port. default: 9999
  -m MODEL, --model MODEL
                        trained model.
  -w WORD_IDX, --word_idx WORD_IDX
                        word to index.
  -t TARGET_IDX, --target_idx TARGET_IDX
                        target to index.

$ bin/dlsuggestd start -m ../model/model_lstm.pth -w ../model/word_to_idx.npy -t ../model/target_to_idx.npy
```

## stop daemon

```bash
$ bin/dlsuggestd close

$ bin/dlsuggestd stop # alias of close
```

## check daemon status

```bash
$ bin/dlsuggestd status
```

# API Reference

Show https://github.com/norihito-hirose/team19/tree/master/atom/dl-suggest/swagger/doc

# example

```bash
$ bin/dlsuggestd start -m ../model/model_lstm.pth -w ../model/word_to_idx.npy -t ../model/target_to_
idx.npy
Loading trained model: ../model/model_lstm.pth
Loading word idx: ../model/word_to_idx.npy
Loading target idx: ../model/target_to_idx.npy
loaded trained model

$ bin/dlsuggestd status
running

$ curl localhost:9999/v1/predict?in=tf.ones | jq . 
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   882    0   882    0     0  78226      0 --:--:-- --:--:-- --:--:-- 88200
{
	"candidates": [
	{
		"probability": 0.24663028120994568,
		"code": "tf.test.mai"
	},
	.......
	{
		"probability": 0.00897204503417015,
		"code": "tf.nn.conv1d"
	}
	],
	"info": {
		"word_to_idx_file": "../model/word_to_idx.npy",
		"model_file": "../model/model_lstm.pth",
		"target_to_idx_file": "../model/target_to_idx.npy",
		"request_url": "/v1/predict?in=tf.ones"
	}
}

```

# Test

```bash
$ python -m unittest discover test
```
