# python3.9
# time:2024/11/19

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence


PRETRAINED_MODEL_NAME = "bert-base-chinese"  
PRETRAINED_MODEL_NAME1 = "bert-base-cased"  

data = pd.read_excel(r'template2word_idf.xlsx')
words = data.values
# print(words)

# tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME1)
bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME1)

tokens = []
for i in range(29):
    tokens.append([tokenizer.tokenize(text=words[i][j]) for j in range(len(words[i]))])
short = []
for i in range(29):
    short.append([tokens[i][j][0] for j in range(len(words[i]))])
print(short)

df = pd.DataFrame(np.array(short))
df.to_excel(r'short_idf.xlsx')
