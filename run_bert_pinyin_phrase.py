# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import json
from pytorch_transformers.modeling_bert_pinyin_phrase import BertForSequenceClassificationPinyinPhrase, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from itertools import cycle
import pypinyin
import gensim
import jieba
jieba.load_userdict('./data/vocab.txt')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassificationPinyinPhrase, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in ( BertConfig,)), ())

logger = logging.getLogger(__name__)

# 计算所有的单词
class words():
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def words_dict(self):
        words_dict = {}
        words_dict['[CLS]'] = 0
        words_dict['[SEP]'] = 1
        words_dict['[unused]'] = 2
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:
                    data = line.strip().split(',')
                    s = [x for x in jieba.cut(data[1] + ' ' + data[2])]
                    for i in s:
                        if i not in words_dict.keys():
                            words_dict[i] = len(words_dict.keys())
        return words_dict

# # 生成初始化向量
# class word2vec():
#     def __init__(self, model_file, words_dict, new_model_file):
#         self.model_file = model_file
#         self.words_dict = words_dict
#         self.new_model_file = new_model_file
#
#     def gen_word2vec(self):
#         nmf = open(self.new_model_file, 'w', encoding='utf-8')
#         model = gensim.models.KeyedVectors.load_word2vec_format(self.model_file, binary=False)
#         keys = model.wv.index2word
#         for key in list(self.words_dict.keys()):
#             if key in keys:
#                 nmf.write(' '.join([str(i) for i in list(model.get_vector(key))]) + '\n')
#             else:
#                 nmf.write(' '.join([str(i) for i in list(model.get_vector(keys[random.randint(0, len(keys))]))]) + '\n')
#         nmf.close()


# 计算所有的拼音
class pinyin():
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_pinyin(self, sentence):
        s = []
        for i in pypinyin.pinyin(sentence, style=pypinyin.NORMAL):
            s.append(i[0])
        return s

    def pinyin_dict(self):
        pinyin_dict = {}
        pinyin_dict['[CLS]'] = 0
        pinyin_dict['[SEP]'] = 1
        pinyin_dict['[unused]'] = 2
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:
                    data = line.strip().split(',')
                    s = self.get_pinyin(data[1] + ' ' + data[2])
                    for i in s:
                        if i not in pinyin_dict.keys():
                            pinyin_dict[i] = len(pinyin_dict.keys())
        return pinyin_dict


def _truncate_str_pair(str_a, str_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(str_a) + len(str_b)
        if total_length <= max_length:
            break
        if len(str_a) > len(str_b):
            str_a = str_a[:-1]
        else:
            str_b = str_b[:-1]
    return str_a, str_b


# 计算words_ids
class words_ids():
    def __init__(self, words_dict, texts, max_seq_len):
        self.words_dict = words_dict
        self.texts = texts
        self.max_seq_len = max_seq_len

    def get_ids(self):
        id_list = []
        for i, text in enumerate(self.texts):
            keys = []
            ids = []
            text_a, text_b = _truncate_str_pair(text.text_a, text.text_b, self.max_seq_len - 3)
            # if i % 1000==0:
            #     print(i)
            keys.append('[CLS]')
            s = []
            for x in jieba.cut(text_a):
                for i in range(len(x)):
                    s.append(x)
            keys.extend(s)
            keys.append('[SEP]')
            s = []
            for x in jieba.cut(text_b):
                for i in range(len(x)):
                    s.append(x)
            keys.extend(s)
            if len(keys) < self.max_seq_len:
                keys.append('[SEP]')
                keys.extend(['[unused]']*(self.max_seq_len-len(keys)))
            else:
                keys = keys[:self.max_seq_len]
                keys.append('[SEP]')
            for k in keys:
                ids.append(self.words_dict.get(k, 2))

            id_list.append(ids)
        return id_list

# 计算pinyin_ids
class pinyin_ids():
    def __init__(self, pinyin_dict, texts, max_seq_len):
        self.pinyin_dict = pinyin_dict
        self.texts = texts
        self.max_seq_len = max_seq_len

    def get_pinyin(self, sentence):
        s = []
        for i in pypinyin.pinyin(sentence, style=pypinyin.NORMAL):
            s.append(i[0])
        return s

    def get_ids(self):
        id_list = []
        for i, text in enumerate(self.texts):
            keys = []
            ids = []
            text_a, text_b = _truncate_str_pair(text.text_a, text.text_b, self.max_seq_len - 3)
            # if i % 1000==0:
            #     print(i)
            keys.append('[CLS]')
            s = self.get_pinyin(text_a)
            keys.extend(s)
            keys.append('[SEP]')
            s = self.get_pinyin(text_b)
            keys.extend(s)
            if len(keys) < self.max_seq_len:
                keys.append('[SEP]')
                keys.extend(['[unused]']*(self.max_seq_len-len(keys)))
            else:
                keys = keys[:self.max_seq_len]
                keys.append('[SEP]')
            for k in keys:
                ids.append(self.pinyin_dict.get(k, 2))

            id_list.append(ids)
        return id_list





class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        
def read_examples(input_file, is_training):
    df=pd.read_csv(input_file, error_bad_lines=False, quoting=csv.QUOTE_NONE)
    examples=[]
    for val in df[['id','sentence1','sentence2','label']].values:
        examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2],label=val[3]))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,split_num,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):

        context_tokens=tokenizer.tokenize(example.text_a)
        ending_tokens=tokenizer.tokenize(example.text_b)
        

        skip_len=len(context_tokens)/split_num  
        choices_features = []
        for i in range(split_num):
            context_tokens_choice=context_tokens[int(i*skip_len):int((i+1)*skip_len)]   
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"]+ ending_tokens + ["[SEP]"] +context_tokens_choice  + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)


            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))


            label = example.label
            if example_index < 1 and is_training:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))


        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels,outputs,labels=[0,1],average='macro')

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--meta_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--lstm_hidden_size", default=300, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="")    
    
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=3, type=int,
                        help="text split")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    file_paths = [os.path.join(args.data_dir, 'train.csv'), os.path.join(args.data_dir, 'dev.csv'), os.path.join(args.data_dir, 'test.csv')]
    pinyin1 = pinyin(file_paths)
    pinyin_dict = pinyin1.pinyin_dict()
    words1 = words(file_paths)
    words_dict = words1.words_dict()
    # print('生成新的词向量')
    # word2vec1 = word2vec('./2000000-small.txt', words_dict, './data/word2vec.txt')
    # word2vec1.gen_word2vec()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    set_seed(args)


    try:
        os.makedirs(args.output_dir)
    except:
        pass
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    
    
    print('pinyin个数：', len(pinyin_dict.keys()))
    print('words个数：', len(words_dict.keys()))

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=2, output_hidden_states=True, pinyin_vocabsize=len(pinyin_dict.keys()), word_vocabsize=len(words_dict.keys()))
    
    # Prepare model
    model = BertForSequenceClassificationPinyinPhrase.from_pretrained(args.model_name_or_path,args,config=config)


        
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.do_train:

        # Prepare data loader
        train_examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training=True)
        print('开始执行拼音')
        pyids = pinyin_ids(pinyin_dict, train_examples, args.max_seq_length)
        pyid_list = pyids.get_ids()
        print('开始执行单词')
        words_ids1 = words_ids(words_dict, train_examples, args.max_seq_length)
        wordid_list = words_ids1.get_ids()

        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length,args.split_num, True)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([int(f.label) for f in train_features], dtype=torch.long)
        print(len(all_label))
        pyid_list = torch.tensor(pyid_list, dtype=torch.long)
        wordid_list = torch.tensor(wordid_list, dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, pyid_list,wordid_list)
        print('结束执行拼音，单词')
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps


        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.train_steps)
        
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_acc=0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0        
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, pyid_list,wordid_list = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, pinyin_ids=None, words_ids=wordid_list)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0 
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))


            if args.do_eval and (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                for file in ['dev.csv']:
                    inference_labels=[]
                    gold_labels=[]
                    inference_logits=[]
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training=True)
                    pyids = pinyin_ids(pinyin_dict, eval_examples, args.max_seq_length)
                    pyid_list = pyids.get_ids()
                    words_ids1 = words_ids(words_dict, eval_examples, args.max_seq_length)
                    wordid_list = words_ids1.get_ids()
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,args.split_num,False)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                    pyid_list = torch.tensor(pyid_list, dtype=torch.long)
                    wordid_list = torch.tensor(wordid_list, dtype=torch.long)

                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, pyid_list,wordid_list)
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)  
                        
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids, pyid_list, wordid_list in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        pyid_list = pyid_list.to(device)
                        wordid_list = wordid_list.to(device)


                        with torch.no_grad():
                            tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, pinyin_ids=None, words_ids=wordid_list)
                            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, pinyin_ids=None, words_ids=wordid_list)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1
                        
                    gold_labels=np.concatenate(gold_labels,0) 
                    inference_logits=np.concatenate(inference_logits,0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = accuracy(inference_logits, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_F1': eval_accuracy,
                              'global_step': global_step,
                              'loss': train_loss}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*'*80)
                        writer.write('\n')
                    if eval_accuracy>best_acc and 'dev' in file:
                        print("="*80)
                        print("Best F1",eval_accuracy)
                        print("Saving Model......")
                        best_acc=eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("="*80)
                    else:
                        print("="*80)
    if args.do_test:
        del model
        gc.collect()
        args.do_train=False
        model = BertForSequenceClassificationPinyinPhrase.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"),args,config=config)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)        
        
        
        for file,flag in [('dev.csv','dev'),('test.csv','test')]:
            inference_labels=[]
            gold_labels=[]
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = False)
            pyids = pinyin_ids(pinyin_dict, eval_examples, args.max_seq_length)
            pyid_list = pyids.get_ids()
            words_ids1 = words_ids(words_dict, eval_examples, args.max_seq_length)
            wordid_list = words_ids1.get_ids()
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,args.split_num,False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
            pyid_list = torch.tensor(pyid_list, dtype=torch.long)
            wordid_list = torch.tensor(wordid_list, dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label, pyid_list,wordid_list)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids, pyid_list, wordid_list in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                pyid_list = pyid_list.to(device)
                wordid_list = wordid_list.to(device)

                with torch.no_grad():
                    # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, pinyin_ids=pyid_list, words_ids=wordid_list).detach().cpu().numpy()
                    logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, pinyin_ids=None, words_ids=wordid_list).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(logits)
                gold_labels.append(label_ids)
            gold_labels=np.concatenate(gold_labels,0)
            logits=np.concatenate(inference_labels,0)
            print(flag, accuracy(logits, gold_labels))
            if flag=='test':
                df=pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0']=logits[:,0]
                df['label_1']=logits[:,1]
                df[['id','label_0','label_1']].to_csv(os.path.join(args.output_dir, "sub.csv"),index=False)
            
if __name__ == "__main__":
    main()

