import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, txt="./dataset2/data0.txt"):
        file = open(txt, 'r')
        lines = file.readlines()
        state = []
        action = []
        temp = ""
        for index, line in enumerate(lines):
            if line.find("[") != -1 and line.find("]") != -1:
                line = line.strip('\n')
                line = line.lstrip('[')
                line = line.split(']')
                line[1] = line[1].lstrip()
                # line[1] = line[1].lstrip('[')
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
                data1 = temp.split() + line[0].split()
                action.append(line[1])
                state.append(data1)

        state = np.array(state)
        state = state.astype(float)
        action = np.array(action)
        action = action.astype(int)
        self.action = action

        self.state = np.reshape(state, [-1, 4, 4])
        file.close()

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]

# for i, sample_batched in enumerate(dataloader):


def load_data(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


if __name__ == "__main__":
    dataset = Dataset(txt="./data0.txt")
    for _, (states, action) in enumerate(load_data(dataset)):
        action = action.cpu().numpy()
        states = states.cpu().numpy()
    print("OK!")
