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


PRETRAINED_MODEL_NAME = "bert-base-chinese"  
PRETRAINED_MODEL_NAME1 = "bert-base-cased"  

data = pd.read_csv(r'HDFS templates changed.csv')
template = data.values[:, 1]
# print(template)

# tokenizer
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
TF-IDF
Parameters:
     list_words:wordd list
Returns:
     dict_feature_select
"""


def feature_select(list_words):
    doc_frequency = defaultdict(int)  
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # compute TF value
    word_tf = {}  # store tf value
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # compute IDF value
    doc_num = len(list_words)
    word_idf = {}  # store idf value
    word_doc = defaultdict(int)  
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i]))

    # compute TF*IDF value
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # descending order
    # dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    dict_idf_select = sorted(word_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_idf_select


if __name__ == '__main__':
    features = feature_select(dataset)  # all TF-IDF value
    print(features)
    print(len(features))
    df = pd.DataFrame(features)
    df.to_excel(r'idf.xlsx')
