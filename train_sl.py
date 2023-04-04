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


from swarmlearning.pyt import SwarmCallback

cache_size = 50
agent_num = 1
agent = []
me = []
modellist = []

default_max_epochs = 5
default_min_peers = 2
# maxEpochs = 2
trainPrint = True
# tell swarm after how many batches
# should it Sync. We are not doing
# adaptiveRV here, its a simple and quick demo run
swSyncInterval = 128

epsilon = 1.0
delta = 1e-5
sigma = 4.0 * (1.0 / epsilon) * \
    torch.sqrt(torch.log(torch.tensor(1.25 / delta)))

dataDir = os.getenv('DATA_DIR', '/platform/data')
scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
modelDir = os.getenv('MODEL_DIR', '/platform/model')
max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
batchSz = 128
useCuda = torch.cuda.is_available()
device = torch.device("cuda" if useCuda else "cpu")


path = os.path.join(dataDir, 'agent.npy')
model_path = os.path.join(dataDir, 'agent.pt')
me.append(np.load(path))
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
agent.load_memory(path)
agent.load_model(model_path)
modellist.append([])
# Create Swarm callback
testDs = me[0]
swarmCallback = None
swarmCallback = SwarmCallback(syncFrequency=swSyncInterval,
                              minPeers=min_peers,
                              useAdaptiveSync=False,
                              adsValData=testDs,
                              adsValBatchSize=batchSz,
                              model=agent.Enet)
# initalize swarmCallback and do first sync
swarmCallback.on_train_begin()

for epoch in range(5):
    for step in range(128):
        for i in range(agent_num):
            agent.learn()
    agent.DP_model(sigma)
    swarmCallback.on_epoch_end(epoch)
    print("syncnum=%d,同步成功" % (epoch))

#     # handles what to do when training ends
swarmCallback.on_train_end()


saved_model_path = os.path.join(scratchDir,  'agent.pt')
for num in range(agent_num):
    agent.save_model(saved_model_path)
