# python3.9
# time:2024/11/13

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
PRETRAINED_MODEL_NAME1 = "bert-base-cased"  # 指定 BERT-BASE 預訓練模型

data = pd.read_csv(r'HDFS templates changed.csv')
template = data.values[:, 1]
# print(template)
short = pd.read_excel(r'short_idf.xlsx')
short = short.values
# print(short)

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME1)
bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME1)

# tokens = tokenizer.encode_plus(text=template[4])
# print(template[4])
# print(tokens)
#
# # 将索引还原成文本
# retokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
# combined_text = " ".join(retokens)
# print(combined_text)

# 把short变为索引
short = [tokenizer.convert_tokens_to_ids(short[i]) for i in range(29)]
# print(short)

# 将29个模板拼接在一起，形成tokens、segments、masks
tokens = [torch.tensor(tokenizer.encode_plus(text=template[i])['input_ids']) for i in range(29)]
tokens = pad_sequence([tokens[i] for i in range(29)], batch_first=True)  # 模板索引向量
print(tokens)
segments = torch.zeros(tokens.shape, dtype=torch.long)
masks = torch.zeros(tokens.shape, dtype=torch.long)
masks = masks.masked_fill(tokens != 0, 1)
# print(tokens[0], segments[0], masks[0])

outputs = bert(tokens, segments, masks)  # torch.Size([29, 33, 768])
print(outputs[0].shape)

# 根据short的索引对应tokens的索引拼接3个词向量
# templatevec = []
# for i in range(29):
#     for j in range(len(short[i])):
#         if j == 0:
#             wordvec = outputs[0][i, list(tokens[i]).index(short[i][j])]
#         else:
#             wordvec = torch.cat([wordvec, outputs[0][i, list(tokens[i]).index(short[i][j])]])
#     # print(wordvec.shape)
#     templatevec.append(np.array(wordvec.detach()))
# # print(templatevec)  # （29，2304）
# # print(len(templatevec))

# 降维============================================
# PCA
# pca = PCA(n_components=20)
# newX = pca.fit_transform(templatevec)
# print(np.array(newX).shape)
# print(sum(pca.explained_variance_ratio_))

# tSNE
# tsne = TSNE(n_components=27, perplexity=27, random_state=42, method="exact")
# X_tsne = tsne.fit_transform(np.array(templatevec))
# print(X_tsne.shape)
# y = range(29)
# # 可视化
# # plt.figure(figsize=(8, 8))
# # for i in range(len(y)):
# #     plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=y[i])
# # plt.legend()
# # plt.show()
#


# df = pd.DataFrame(np.array(newX))
# df.to_excel(r'PCA20_idf.xlsx')  # 20:0.97 16：0.94