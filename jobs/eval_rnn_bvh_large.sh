#!/bin/bash
echo "Running RNN on [...] Dataset"
python ./models/rnn/main.py --task dummy --model_dir rnn_dummy --model_type BLSTM --num_train_epochs 10 --train_batch_size 512 --eval_batch_size 512 --max_seq_len 200 --do_eval --logging_steps 5000 --save_steps 5000
echo "Running RNN on [...] Dataset"
python ./models/rnn/main.py --task dummy_masked --model_dir rnn_dummy_masked --model_type BLSTM --num_train_epochs 10 --train_batch_size 512 --eval_batch_size 512 --max_seq_len 200 --do_eval --logging_steps 5000 --save_steps 5000
echo "Running RNN on [...] Dataset"
python ./models/rnn/main.py --task dummy_aug --model_dir rnn_dummy_aug --model_type BLSTM --num_train_epochs 10 --train_batch_size 512 --eval_batch_size 512 --max_seq_len 200 --do_eval --logging_steps 5000 --save_steps 5000
echo "Running RNN on [...] Dataset"
python ./models/rnn/main.py --task dummy_masked_aug --model_dir rnn_dummy_masked_aug --model_type BLSTM --num_train_epochs 10 --train_batch_size 512 --eval_batch_size 512 --max_seq_len 200 --do_eval --logging_steps 5000 --save_steps 5000