# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2019/11/13 10:20
# -*- python3.6
# coding=utf-8
import sys
from xml.dom.minidom import parse
from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd


def generate_train_data_pair(equ_questions, not_equ_questions):
    a = [str(x).replace(',', '，') + "," + str(y).replace(',', '，') + "," + "0" for x in equ_questions for y in not_equ_questions if x != y]
    b = [str(x).replace(',', '，') + "," + str(y).replace(',', '，') + "," + "1" for x in equ_questions for y in equ_questions if x != y]
    c = [str(x).replace(',', '，') + "," + str(y).replace(',', '，') + "," + "0" for x in not_equ_questions for y in not_equ_questions if x != y]
    return a + b


def parse_train_data(xml_data):
    pair_list = []
    doc = parse(xml_data)
    collection = doc.documentElement
    for i in collection.getElementsByTagName("Questions"):
        # if i.hasAttribute("number"):
        #     print ("Questions number=", i.getAttribute("number"))
        EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
        NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
        equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
        not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
        equ_questions_list, not_equ_questions_list = [], []
        for q in equ_questions:
            try:
                equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        for q in not_equ_questions:
            try:
                not_equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        pair = generate_train_data_pair(equ_questions_list, not_equ_questions_list)
        pair_list.extend(pair)
    print("All pair count=", len(pair_list))
    return pair_list


def write_train_data(file, pairs):
    with open(file, "w", encoding='utf-8') as f:
        for pair in pairs:
            f.write(pair + "\n")


def gen_train_dev(file, fw_file, guid=0):
    fw = open(fw_file, 'w', encoding='utf-8')
    fw.write('id,content,title,label\n')
    for line in file:
        data = line.strip().split(',')
        guid = guid + 1
        fw.write(str(guid) + ',' + str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + '\n')

    fw.close()
    return guid


def gen_test(test, fw_file):
    fw = open(fw_file, 'w', encoding='utf-8')
    fw.write('id,content,title,label\n')
    with open(test, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()[1:]):
            fw.write(str(i) + ',' + line)
    fw.close()


# 生成交叉所需要的样本
def k_flod(file, test_file, k=5):
    with open(file, 'r', encoding='utf-8') as f:
        temp = np.array(f.readlines())
        for i in range(k):
            file_path = 'data_' + str(i)
            if not os.path.exists(file_path):
                os.mkdir(file_path)
        kf = KFold(n_splits=k)
        np.random.shuffle(temp)
        num = 0
        for train_index, dev_index in kf.split(temp):
            X_train, X_dev = temp[train_index], temp[dev_index]
            guid2 = gen_train_dev(X_train, 'data_' + str(num) + '/train.csv', guid=0)
            gen_train_dev(X_dev, 'data_' + str(num) + '/dev.csv', guid=guid2)
            gen_test(test_file, 'data_' + str(num) + '/test.csv')
            num = num + 1

def static_data(file):
    len_result = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            len_result.append(len(data[0]))
            len_result.append(len(data[2]))

    # len_result = list(set(len_result))
    print((pd.DataFrame(len_result)).describe())

def gen_train_dev2(file, fw_file, guid=0):
    fw = open(fw_file, 'w', encoding='utf-8')
    # fw.write('id,content,title,label\n')
    for line in file:
        data = line.strip().split(',')
        guid = guid + 1
        fw.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\n')

    fw.close()
    return guid


def gen_test2(test, fw_file):
    fw = open(fw_file, 'w', encoding='utf-8')
    # fw.write('id,content,title,label\n')
    with open(test, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()[1:]):
            # fw.write('\t'.join(line.split(',')))
            data = line.split(',')
            fw.write(str(data[0]) + '\t' + str(data[1]) + '\t1\n')
    fw.close()

# 生成交叉所需要的样本
def k_flod2(file, test_file, k=5):
    with open(file, 'r', encoding='utf-8') as f:
        temp = np.array(f.readlines())
        for i in range(k):
            file_path = 'data_' + str(i)
            if not os.path.exists(file_path):
                os.mkdir(file_path)
        kf = KFold(n_splits=k)
        np.random.shuffle(temp)
        num = 0
        for train_index, dev_index in kf.split(temp):
            X_train, X_dev = temp[train_index], temp[dev_index]
            guid2 = gen_train_dev2(X_train, 'data_' + str(num) + '/train.csv', guid=0)
            gen_train_dev2(X_dev, 'data_' + str(num) + '/dev.csv', guid=guid2)
            gen_test2(test_file, 'data_' + str(num) + '/test.csv')
            num = num + 1

def gen_train_dev3(file, fw_file, guid=0):
    fw = open(fw_file, 'w', encoding='utf-8')
    fw.write('id,content,title,label\n')
    for line in file:
        data = line.strip().split(',')
        guid = guid + 1
        fw.write(str(guid) + ',' + str(data[0]).replace(',', '，') + ',' + str(data[2]).replace(',', '，') + ',' + str(data[1]) + '\n')

    fw.close()
    return guid


def gen_test3(test, fw_file):
    fw = open(fw_file, 'w', encoding='utf-8')
    fw.write('id,content,title,label\n')
    with open(test, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()[1:]):
            fw.write(str(i) + ',' + line)
    fw.close()


# 生成交叉所需要的样本
def k_flod3(file, test_file, k=5):
    with open(file, 'r', encoding='utf-8') as f:
        temp = np.array(f.readlines()[1:])
        for i in range(k):
            file_path = 'data_' + str(i)
            if not os.path.exists(file_path):
                os.mkdir(file_path)
        kf = KFold(n_splits=k)
        np.random.shuffle(temp)
        num = 0
        for train_index, dev_index in kf.split(temp):
            X_train, X_dev = temp[train_index], temp[dev_index]
            guid2 = gen_train_dev3(X_train, 'data_' + str(num) + '/train.csv', guid=0)
            gen_train_dev3(X_dev, 'data_' + str(num) + '/dev.csv', guid=guid2)
            gen_test3(test_file, 'data_' + str(num) + '/test.csv')
            num = num + 1

if __name__ == "__main__":
    pair_list = parse_train_data("./train_set.xml")
    write_train_data("./train_data.txt", pair_list)
    k_flod('train_data.txt', 'test.csv', k=5)
    # static_data('train_data.txt')
    # k_flod2('train_data.txt', 'test.csv', k=5)
    # gen_test2('./test.csv', './test2.csv')
    # k_flod3('train_eda.csv', 'test.csv', k=5)

