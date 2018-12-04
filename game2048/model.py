import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def resnet_block(input_size, num_filters, kernel_size=3, stride=1, activation=None):
    if activation is not None:
        output = nn.Sequential(
            nn.Conv2d(input_size, num_filters, kernel_size=kernel_size, stride=stride),
            #nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
    else:
        output = nn.Sequential(
            nn.Conv2d(input_size, num_filters, kernel_size=kernel_size, stride=stride),
            #nn.BatchNorm2d(num_filters)
        )
    return output


class Conv_block(nn.Module):
    def __init__(self, state_size, action_size):
        super(Conv_block, self).__init__()
        self.num_actions = action_size
        self.conv1 = nn.Conv2d(state_size, 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_size)
        )
        self.fc1_adv = nn.Linear(in_features=2048, out_features=256)
        self.fc1_val = nn.Linear(in_features=2048, out_features=256)

        self.fc2_adv = nn.Linear(in_features=256, out_features=action_size)
        self.fc2_val = nn.Linear(in_features=256, out_features=1)

        #self.fc1_adv = nn.Linear(in_features=2048, out_features=512)
        #self.fc1_val = nn.Linear(in_features=2048, out_features=512)

        #self.fc2_adv = nn.Linear(in_features=512, out_features=action_size)
        #self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.resnet_conv_0 = nn.Conv2d(state_size, 128, kernel_size=1)
        self.resnet_block1_1 = resnet_block(128, 256, kernel_size=2, activation="relu")
        self.resnet_block1_2 = resnet_block(256, 256, kernel_size=1)
        self.resnet_conv_1 = nn.Conv2d(128, 256, kernel_size=2)
        self.resnet_block2_1 = resnet_block(256, 512, kernel_size=2, activation="relu")
        self.resnet_block2_2 = resnet_block(512, 512, kernel_size=1)
        self.resnet_conv_2 = nn.Conv2d(256, 512, kernel_size=2)

    """
    def forward(self, x):
        # x : [batch_size, 16, 4, 4]
        x = self.conv1(x)
        #x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #x = self.batch_norm3(x)
        x = x.view(-1, 1024)
        adv = self.fc1_adv(x)
        val = self.fc1_val(x)
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        x = F.log_softmax(x)
        return x

    """
    def forward(self, x):
        x = self.resnet_conv_0(x)
        x = self.relu(x)
        a1 = self.resnet_block1_1(x)
        b1 = self.resnet_block1_2(a1)
        x = self.resnet_conv_1(x)
        x = x + b1
        x = self.relu(x)
        a2 = self.resnet_block2_1(x)
        b2 = self.resnet_block2_2(a2)
        x = self.resnet_conv_2(x)
        x = x + b2
        x = self.relu(x)
        x = x.view(-1, 2048)
        #x = x.view(-1, 1024)
        adv = self.fc1_adv(x)
        val = self.fc1_val(x)
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        x = F.log_softmax(x)
        return x


    def get_weights(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            params[k] = p
        return params

    def copy_weights(self, weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in weights.keys():
                p.data = weights[k].data.clone()