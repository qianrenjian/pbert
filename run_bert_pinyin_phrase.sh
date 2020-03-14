#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
for((i=0;i<1;i++));  
do   

python3.5 run_bert_pinyin_phrase.py \
--model_type bert \
--model_name_or_path chinese_wwm_pytorch \
--do_train \
--do_eval \
--do_test \
--data_dir ./data/lcqmc \
--output_dir ./pp_bert$i \
--max_seq_length 72 \
--split_num 1 \
--lstm_hidden_size 128 \
--lstm_layers 1 \
--lstm_dropout 0.2 \
--eval_steps 1000 \
--per_gpu_train_batch_size 5 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0.003 \
--train_steps 10000

done  





