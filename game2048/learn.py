import torch
from torch.autograd import Variable
import sys
import os
import itertools
import numpy as np
import random
from collections import namedtuple
from utils import *
import torch.optim as optim
from model import Conv_block
from Datasets import *
import time


class Conv_learn:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=10000,
                 batch_size=8,
                 dueling=True,
                 double_q=True
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = e_greedy
        self.dueling = dueling  # decide to use dueling DQN or not

        self.double_q = double_q  # decide to use double q or not

        self.learn_step_counter = 0

        self.memory = Memory(capacity=memory_size)

        self.net = Conv_block(self.n_features, self.n_actions)

        self.cost_his = []

    def get_eval_weight(self):
        params = self.eval_net.get_weights()
        return params  # OrderDict type

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def choose_action(self, observation):
        observation = observation.reshape(4, 16)
        observation = np.expand_dims(observation, axis=0)
        # print(observation.shape)
        observation = torch.cuda.FloatTensor(observation)
        self.eval_net.eval()
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(observation)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            weights = self.get_eval_weight()
            self.target_net.copy_weights(weights)
            print('\ntarget_params_replaced\n')
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        self.target_net.eval()
        self.eval_net.eval()
        s_ = np.zeros([batch_memory.shape[0], 64])
        s_[:, 0:48] = batch_memory[:, 16:64]
        s_[:, 48:64] = batch_memory[:, -16:]
        s_ = s_.reshape(batch_memory.shape[0], 4, 16)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, 64].astype(int)
        q_next = self.target_net.forward(torch.cuda.FloatTensor(s_)).detach()
        q_eval = self.eval_net.forward(torch.cuda.FloatTensor(batch_memory[:, 0:64].reshape(batch_memory.shape[0], s_.shape[1], s_.shape[2])))
        a_index = torch.cuda.LongTensor(np.expand_dims(eval_act_index, axis=1))
        q_eval = q_eval.gather(1, a_index)
        q_target = q_eval.clone()

        reward = torch.cuda.FloatTensor(batch_memory[:, 65]).view(self.batch_size, 1)
        q_target = reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        self.eval_net.train()
        self.abs_errors = torch.mean(torch.abs(q_target - q_eval), dim=1).detach()
        self.loss = torch.mean(torch.cuda.FloatTensor(ISWeights) * (q_target - q_eval) * (q_target - q_eval))
        optimizer = optim.Adam(self.eval_net.parameters())
        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()
        self.abs_errors = self.abs_errors.cpu().numpy()
        self.memory.batch_update(tree_idx, self.abs_errors)
        self.cost_his.append(self.loss)
        self.learn_step_counter += 1

    def save(self):
        torch.save(self.eval_net, 'eval_net.pkl')
        torch.save(self.target_net, 'target_net.pkl')

    def restore(self):
        self.target_net = torch.load('target_net.pkl')
        self.eval_net = torch.load('eval_net.pkl')








