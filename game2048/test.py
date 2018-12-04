import numpy as np

np.random.seed(0)

file = open('./data.txt', mode='w')

x = np.random.randint(0, 100, 16)
a = np.random.randint(0, 4, 1)

for _ in range(16):
    print(x, " ", a, file=file)
