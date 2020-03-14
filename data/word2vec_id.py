# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2020/1/3 10:07
# -*- python3.6
import gensim
import jieba
import numpy as np
import tqdm


# 读取模型文件，将相关vec转化成id对应形式
def wordid_vector(model_file, new_model_file, dict_file, embedding_size=200):
    """
    :param model_file: 模型文件
    :param new_model_file: 新模型文件
    :param dict_file: 对应的字典文件
    :return: id类型模型文件
    """
    dict_result = {}
    nmf = open(new_model_file, 'w', encoding='utf-8')
    df = open(dict_file, 'w', encoding='utf-8')
    nmf.write('100003 200\n')
    nmf.write(' '.join([str(i) for i in list(np.random.normal(0, 0.1, embedding_size))]) + '\n')
    nmf.write(' '.join([str(i) for i in list(np.random.normal(0, 0.1, embedding_size))]) + '\n')
    nmf.write(' '.join([str(i) for i in list(np.random.normal(0, 0.1, embedding_size))]) + '\n')
    dict_result['[CLS]'] = 0
    dict_result['[SEP]'] = 1
    dict_result['[unused]'] = 2
    df.write('[CLS] 0\n[SEP] 1\n[unused] 2\n')

    with open(model_file, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f.readlines()[1:]):
            data = line.strip().split(' ')
            if data[0] not in dict_result.keys():
                dict_result[data[0]] = len(dict_result.keys())
                df.write(str(data[0]) + ' ' + str(len(dict_result.keys())) + '\n')
                nmf.write(' '.join(data[1:]) + '\n')
    df.close()
    nmf.close()


if __name__ == '__main__':
    model_file = 'E:\\模型与语料\\000语言模型\\2000000-small.txt'
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    wordid_vector(model_file, './word2vec.txt', './vocab_dict.txt', embedding_size=200)