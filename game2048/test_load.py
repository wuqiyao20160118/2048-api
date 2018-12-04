import numpy as np
import pandas as pd

txt = "./data0.txt"

file = open(txt, 'r')
lines = file.readlines()
data_size = len(lines)
state = []
action = []
index = 0
temp = ""
for index, line in enumerate(lines):
    if line.find("[") != -1 and line.find("]") != -1:
        line = line.strip('\n')
        line = line.lstrip('[')
        line = line.split(']')
        line[1] = line[1].lstrip()
        #line[1] = line[1].lstrip('[')
        data1 = line[0].split()
        action.append(line[1])
        state.append(data1)
    elif line.find("[") != -1 and line.find("]") == -1:
        line = line.strip('\n')
        line = line.lstrip('[')
        temp = line
    elif line.find("[") == -1 and line.find("]") != -1:
        line = line.strip('\n')
        line = line.split(']')
        line[1] = line[1].lstrip()
        line[1] = line[1].rstrip()
        data1 = temp.split() + line[0].split()
        action.append(line[1])
        state.append(data1)

a = np.array(state)
a = a.astype(float)
b = np.array(action)
b = b.astype(int)
length = a.shape[0]
#print(a)
max_score = a.max(axis=1)
score_frame = pd.DataFrame(max_score)
print(score_frame.apply(pd.value_counts))
#print(b)
file.close()