# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2019/12/10 17:49
# -*- python3.6
import pypinyin
import os
import jieba
from pytorch_pretrained_bert import BertModel
import pandas as pd

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
            tokens_a = tokens_a[:-1]
        else:
            tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b

# 计算pinyin_ids
class pinyin_ids():
    def __init__(self, pinyin_dict, text, max_seq_len):
        self.pinyin_dict = pinyin_dict
        self.text = text
        self.max_seq_len = max_seq_len


    def get_pinyin(self, sentence):
        s = []
        for i in pypinyin.pinyin(sentence, style=pypinyin.NORMAL):
            s.append(i[0])
        return s

    def get_ids(self):
        keys = []
        ids = []
        text_a = '中国真伟大'
        text_b = '中国真美丽'
        text_a, text_b = _truncate_seq_pair(text_a, text_b, self.max_seq_len - 3)
        print(text_a, text_b)
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
        return ids


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
            text_a = '中国要可'
            text_b = '中国赃款'
            text_a, text_b = _truncate_str_pair(text_a, text_b, self.max_seq_len - 3)
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

def read_examples(input_file, is_training=True):
    df=pd.read_csv(input_file)
    examples=[]
    for i, val in enumerate(df[['content','label']].values):
        print(i, val[0], val[1])
    return examples

if __name__ == '__main__':
    # file_paths = [os.path.join('./data/data_0/', 'train.csv'), os.path.join('./data/data_0/', 'dev.csv'), os.path.join('./data/data_0/', 'test.csv')]
    # file_paths = [os.path.join('./data/data_0/', 'test.csv')]
    # word = words(file_paths)
    # word_dict = word.words_dict()
    # print(type(word_dict), word_dict)
    # print(words_ids(word_dict, '1', 10).get_ids())
    read_examples('./data/test.csv')