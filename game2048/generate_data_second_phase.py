from agents import ExpectiMaxAgent
from game import *
import numpy as np

GAME_SIZE = 4
SCORE_TO_WIN = 2048
eposide = 400
game_train = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
agent = ExpectiMaxAgent(game_train)
txt_dir = "./new_data/data400.txt"

index = 400
file_index = 0
file = open(txt_dir, mode='w')
for ep in range(eposide):
    file_dir = "./my_data/data" + str(file_index) + ".txt"
    data_file = open(file_dir, mode="r")
    line = data_file.readline()
    index += 1
    txt_dir = "./new_data/data" + str(index) + ".txt"
    file.close()
    file = open(txt_dir, mode='w')
    temp = ""
    while line:
        flag = 0
        if line.find("[") != -1 and line.find("]") != -1:
            line = line.strip('\n')
            line = line.lstrip('[')
            line = line.split(']')
            # action = line[1].lstrip()  # action
            # line[1] = line[1].lstrip('[')
            state = line[0].split()  # state
            flag = 1
        elif line.find("[") != -1 and line.find("]") == -1:
            line = line.strip('\n')
            line = line.lstrip('[')
            temp = line
            flag = 0
        elif line.find("[") == -1 and line.find("]") != -1:
            line = line.strip('\n')
            line = line.split(']')
            # action = line[1].lstrip()
            state = temp.split() + line[0].split()
            flag = 1
        if flag == 1:
            state = np.array(state)
            state = state.astype(float)
            state = np.reshape(state, [4, 4])
            game_train.set_state(state)
            state = game_train.board
            max_score = np.max(state)
            state_print = np.reshape(state, [1, 16]).squeeze()
            action = agent.step()
            game_train.move(action)
            for _ in range(4):
                print(state_print, " ", action, file=file)
                state = np.rot90(state)
                action = (action + 1) % 4
                state_print = np.reshape(state, [1, 16]).squeeze()
        line = data_file.readline()
    data_file.close()
    file_index += 1
