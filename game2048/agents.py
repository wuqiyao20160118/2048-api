import numpy as np
#from learn_dqn import *
from model import Conv_block
from game import Game
import matplotlib as mpl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from math import log2
from Datasets import *
import random

random.seed(0)


def trans_num(state):
    # state : 4*4 numpt array
    #state_temp = state.reshape(1, 16).squeeze()
    state_temp = state.reshape(1, 16).squeeze()
    #result_state = np.zeros((16, 12))
    result_state = np.zeros((16, 12))
    for i in range(16):
        num = int(state_temp[i])
        length = len(bin(num))
        index = length - bin(num).find('1') - 1
        result_state[i, index] = 1
    result_state = result_state.reshape(1, 16*12).squeeze()
    return result_state


def trans_state(state):
    state_temp = state.reshape(1, 16).squeeze()
    result_state = np.zeros(16)
    for i in range(16):
        temp = int(state_temp[i])
        if temp < 2:
            result_state[i] = 0.0
        else:
            result_state[i] = log2(temp) / 11
    return result_state


def make_input(grid):
    g0 = grid
    #table = {2 ** i: i for i in range(1, 16)}
    table = {2 ** i: i for i in range(1, 12)}
    table[0] = 0
    r = np.zeros(shape=(16, 4, 4), dtype=float)
    #r = np.zeros(shape=(12, 4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            v = int(g0[i, j])
            r[table[v], i, j] = 1
    return r

def trans_input(grid):
    batch_size = grid.shape[0]
    r = np.zeros(shape=(batch_size, 16, 4, 4), dtype=int)
    #r = np.zeros(shape=(batch_size, 12, 4, 4), dtype=float)
    for index in range(batch_size):
        r[index] = np.expand_dims(make_input(grid[index].squeeze()), axis=0)
    return r


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False, type=None):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if type is not None:
                observation_, reward, done = self.game.move(direction)
            else:
                self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

    def collect_memory(self):
        state_size = 16
        action = self.step()
        state = self.game.board
        state = trans_state(state)
        state = np.reshape(state, [1, state_size])
        next_state, reward, done = self.game.move(action, 'EMA')
        next_state = trans_state(next_state)
        next_state = np.reshape(next_state, [1, state_size])
        return state, reward, action, next_state, done

class ConvAgent(Agent):

    def __init__(self, game, display=None):
        self.load = False
        self.game = game
        self.state_size = 16
        #self.state_size = 12
        self.action_size = 4
        self.model = Conv_block(self.state_size, self.action_size).cuda()
        self.model1 = Conv_block(self.state_size, self.action_size).cuda()
        self.model2 = Conv_block(self.state_size, self.action_size).cuda()
        self.model3 = Conv_block(self.state_size, self.action_size).cuda()
        self.model4 = Conv_block(self.state_size, self.action_size).cuda()
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

    def get_action(self, state, model=None):
        # epsilon explorer
        if model is None:
            model = self.model

        state = torch.from_numpy(state).cuda()
        state = Variable(state).float()
        q_value = model(state)
        _, action = torch.max(q_value, 1)
        return int(action)

    def train(self):
        EPISODES = 40
        env = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
        #game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
        #state_size = 16
        #action_size = 4

        self.learning_rate = 0.0001

        #self.agent = DQNAgent(state_size, action_size)
        #self.expect_max = ExpectiMaxAgent(game)
        #self.agent_max = DQNAgent(state_size, action_size, memory_size=50000)

        scores, episodes = [], []
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss()
        episode = 0
        ep = -1
        max_score = 0
        range_list = range(0, 801)
        ran_range = random.sample(range_list, 801)
        # score_threshold = 32

        while episode <= EPISODES:
            if episode <= 15:
                self.learning_rate = 0.001
            else:
                self.learning_rate = 0.0001
            ep += 1
            if ep > 800:
                ep = 0
                episode += 1
                total_score = 0
                for j in range(10):
                    state = env.reset()
                    state = make_input(state)
                    state = np.expand_dims(state, axis=0)
                    while env.end == 0:
                        # get action for the current state and go one step in environment
                        action = self.get_action(state)
                        env.move(action)
                        state = env.board
                        state = make_input(state)
                        state = np.expand_dims(state, axis=0)
                    score = env.final_score()
                    total_score += score
                    print("episode:", episode, "  score:", score)
                if total_score > max_score:
                    torch.save(self.model, "./2048_dqn_2nd" + str(episode))
                ran_range = random.sample(range_list, 801)
            r = ran_range[ep]
            data_dir = "./new_data/data" + str(r) + ".txt"
            dataset = Dataset(txt=data_dir)
            for _, (states, action) in enumerate(load_data(dataset)):
                action = action.cpu().numpy()
                states = trans_input(states)
                states = torch.cuda.FloatTensor(states)
                states = Variable(states).float()
                action = torch.cuda.LongTensor(action)
                action = Variable(action)
                prediction = self.model(states)
                loss = loss_function(prediction, action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #torch.save(self.model, "./2048_dqn_conv4_" + str(episode))

    def fine_tune(self):
        EPISODES = 40
        env = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
        # game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
        # state_size = 16
        # action_size = 4

        self.learning_rate = 0.0001

        # self.agent = DQNAgent(state_size, action_size)
        # self.expect_max = ExpectiMaxAgent(game)
        # self.agent_max = DQNAgent(state_size, action_size, memory_size=50000)

        scores, episodes = [], []
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss()
        episode = 0
        ep = -1
        max_score = 0
        range_list = range(0, 400)
        ran_range = random.sample(range_list, 400)
        agent = ConvAgent(env)
        # score_threshold = 32

        while episode <= EPISODES:
            if episode <= 15:
                self.learning_rate = 0.001
            else:
                self.learning_rate = 0.0001
            ep += 1
            if ep > 399:
                ep = 0
                episode += 1
                total_score = 0
                for j in range(10):
                    state = env.reset()
                    state = make_input(state)
                    state = np.expand_dims(state, axis=0)
                    agent.model = torch.load('./2048_dqn_conv5_15')
                    while env.end == 0:
                        if agent.game.score <= 32:
                            action = agent.get_action(state)
                            env.move(action)
                            state = env.board
                            state = make_input(state)
                            state = np.expand_dims(state, axis=0)
                        else:
                            # get action for the current state and go one step in environment
                            action = self.get_action(state)
                            env.move(action)
                            state = env.board
                            state = make_input(state)
                            state = np.expand_dims(state, axis=0)
                    score = env.final_score()
                    total_score += score
                    print("episode:", episode, "  score:", score)
                if total_score > max_score:
                    torch.save(self.model, "./2048_dqn_2nd_ep" + str(episode))
                ran_range = random.sample(range_list, 400)
            r = ran_range[ep]
            data_dir = "./dataset_new/data" + str(r) + ".txt"
            dataset = Dataset(txt=data_dir)
            for _, (states, action) in enumerate(load_data(dataset)):
                action = action.cpu().numpy()
                states = trans_input(states)
                states = torch.cuda.FloatTensor(states)
                states = Variable(states).float()
                action = torch.cuda.LongTensor(action)
                action = Variable(action)
                prediction = self.model(states)
                loss = loss_function(prediction, action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def step(self):
        count = [0, 0, 0, 0]
        if not self.load:
            self.model = torch.load('./2048_dqn_2nd1')
            self.model1 = torch.load('./2048_dqn_2nd2')
            self.model2 = torch.load('./2048_dqn_2nd3')
            self.load = True

        #state = np.reshape(self.game.board, [1, 16]).squeeze()

        observation = make_input(self.game.board)
        observation = np.expand_dims(observation, axis=0)

        board1 = np.rot90(self.game.board)
        board2 = np.rot90(board1)
        board3 = np.rot90(board2)
        observation1 = make_input(board1)
        observation2 = make_input(board2)
        observation3 = make_input(board3)
        observation1 = np.expand_dims(observation1, axis=0)
        observation2 = np.expand_dims(observation2, axis=0)
        observation3 = np.expand_dims(observation3, axis=0)

        direction = self.get_action(observation)
        count[direction] += 1
        direction1 = self.get_action(observation, model=self.model1)
        count[direction1] += 1
        direction2 = self.get_action(observation, model=self.model2)
        count[direction2] += 1

        direction = self.get_action(observation1)
        count[(direction+3) % 4] += 1
        direction1 = self.get_action(observation1, model=self.model1)
        count[(direction1 + 3) % 4] += 1
        direction2 = self.get_action(observation1, model=self.model2)
        count[(direction2 + 3) % 4] += 1

        direction = self.get_action(observation2)
        count[(direction + 2) % 4] += 1
        direction1 = self.get_action(observation2, model=self.model1)
        count[(direction1 + 2) % 4] += 1
        direction2 = self.get_action(observation2, model=self.model2)
        count[(direction2 + 2) % 4] += 1

        direction = self.get_action(observation3)
        count[(direction + 1) % 4] += 1
        direction1 = self.get_action(observation3, model=self.model1)
        count[(direction1 + 1) % 4] += 1
        direction2 = self.get_action(observation3, model=self.model2)
        count[(direction2 + 1) % 4] += 1

        direction = count.index(max(count))

        return direction


if __name__ == "__main__":
    # CUDA variables
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print('Using GPU!')
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    game_train = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
    agent = ConvAgent(game_train)
    agent.train()
    #agent.fine_tune()
