# python3.9
# time:2025/8/5

import random
import scipy.stats as ss
import numpy as np
import pandas as pd

# 0:cnncosdotproduct 1:cnnVaswani 2:cnnnoattn 3:grucosdotproduct 4:gruVaswani 5:grunoattn
books1 = pd.read_excel(r'BGLF1.xlsx')
data = books1.values
# print(data)
x = data[200:, 0]
y = data[200:, 2]
print(x.shape)
# x = [random.random() for i in range(8)]
# x = [0.16910235172664467, 0.2020437094569132, 0.6924720854383165, 0.5262249846925307, 0.7309038310126759, 0.972498016667232, 0.6576645372688086, 0.6169914457716444]

# y = [random.random() + 5 for i in range(10)]
# y = [5.060202647284982, 5.7298963211653895, 5.0386386764253155, 5.625106039575542, 5.962911362302517, 5.731971211803443, 5.4344784457821005, 5.6345062250337, 5.847658639360786, 5.7380763398004735]

stats1, p1 = ss.ranksums(x, y, alternative='two-sided')  # H0:x=y
stats2, p2 = ss.ranksums(x, y, alternative='greater')  # H0:x<y
stats3, p3 = ss.ranksums(x, y, alternative='less')  # H0:x>y

print('p1:', p1)
print('p2:', p2)
print('p3:', p3)