# python3.9
# time:2024/11/19

import numpy as np
import pandas as pd
from collections import defaultdict

data = pd.read_csv(r'HDFS templates changed.csv')
template = data.values[:, 1]
template = [x.lower() for x in template]
word = pd.read_excel('idf.xlsx')
word = word.values
# print(template)
dataset = []
for i in range(29):
    dataset.append(template[i].split(" "))
# print(word)
words = dict()
for i in range(len(dataset)):
    num = 0
    words[i] = []
    for j in range(len(word)):
        if word[j, 0] in dataset[i]:
            num += 1
            words[i].append(word[j, 0])
        if num > 2:
            break
print(words)
df = pd.DataFrame(words)
df.to_excel(r'template2word_idf.xlsx')