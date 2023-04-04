import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from cache.DataLoader import DataLoaderZipfpdf
from agents.CacheAgent import *
from agents.DQNAgent_p4 import DQNAgent

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import collections

from agents.CacheAgent import LearnerAgent, RandomAgent, LRUAgent, LFUAgent

# from swarmlearning.pyt import SwarmCallback

cache_size = 50
agent_num = 2
agent = []
me = []
modellist = []

for i in range(agent_num):
    path = os.path.join('./data/agent'+str(i), 'agent.npy')
    model_path = os.path.join(
        './data/agent'+str(i), 'agent.pt')
    agent.append(DQNAgent(cache_size+1, cache_size*5+3,
                          learning_rate=0.01,
                          reward_decay=0,

                          # Epsilon greedy
                          e_greedy_min=(0.0, 0.1),
                          e_greedy_max=(0.2, 0.8),
                          e_greedy_init=(0.1, 0.5),
                          e_greedy_increment=(0.005, 0.01),
                          e_greedy_decrement=(0.005, 0.001),

                          history_size=50,
                          dynamic_e_greedy_iter=25,
                          reward_threshold=3,
                          explore_mentor='LRU',

                          replace_target_iter=100,
                          memory_size=10000,
                          batch_size=128,

                          output_graph=False,
                          #   privacy=True,
                          verbose=0
                          ))
    agent[i].load_memory(path)
    agent[i].load_model(model_path)
    modellist.append([])

for epoch in range(5):
    for step in range(5):
        for i in range(agent_num):
            agent[i].learn()

    for num in range(agent_num):
        modellist[num] = agent[num].Enet.state_dict()
    weights_keys = list(modellist[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weights_keys:
        key_sum = 0
        for i in range(len(modellist)):
            key_sum = key_sum+modellist[i][key]
        fed_state_dict[key] = key_sum/len(modellist)

    for num in range(agent_num):
        agent[num].Enet.load_state_dict(fed_state_dict)
        agent[num].Tnet.load_state_dict(fed_state_dict)
    print("syncnum=%d,同步成功" % (epoch))

for num in range(agent_num):
    path = os.path.join('./data/agent'+str(i), 'agent.npy')
    model_path = os.path.join(
        './data/agent'+str(i), 'agent.pt')
    agent[i].save_memory(path)
    agent[i].save_model(model_path)
