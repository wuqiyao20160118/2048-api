from agents import ExpectiMaxAgent
from game import *
import numpy as np

GAME_SIZE = 4
SCORE_TO_WIN = 2048
eposide = 4000
game_train = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
agent = ExpectiMaxAgent(game_train)
txt_dir = "./dataset2/data0.txt"
index = 0
file = open(txt_dir, mode='w')
for ep in range(eposide):
    _ = game_train.reset()
    if ep % 10 == 0 and ep != 0:
        index += 1
        txt_dir = "./dataset2/data" + str(index) + ".txt"
        file.close()
        file = open(txt_dir, mode='w')
    while game_train.end == 0:
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
        """
        if max_score >= 16:
            max_score = 1024 / max_score
        else:
            max_score = 64
        while max_score != 0:
            print(state, " ", action, file=file)
            max_score -= 1
        """


