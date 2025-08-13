# python3.9
# time:2024/11/19

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import math
import operator


PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
PRETRAINED_MODEL_NAME1 = "bert-base-cased"  # 指定 BERT-BASE 預訓練模型

data = pd.read_csv(r'HDFS templates changed.csv')
template = data.values[:, 1]
# print(template)

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME1)
# bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME1)

# tokens = tokenizer.encode_plus(text=template[0])
# tokens = tokenizer.tokenize(text=template[0])
# print(template[0])
# print(tokens)

template = [x.lower() for x in template]
dataset = []
for i in range(29):
    dataset.append(template[i].split(" "))
# print(dataset)

"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""


def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)  # 全文档各单词出现次数
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储每个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i]))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # 对字典按值由大到小排序
    # dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    dict_idf_select = sorted(word_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_idf_select


if __name__ == '__main__':
    features = feature_select(dataset)  # 所有词的TF-IDF值
    print(features)
    print(len(features))
    df = pd.DataFrame(features)
    df.to_excel(r'idf.xlsx')
