"""
This code is partially from Morvan Zhou
https://morvanzhou.github.io/tutorials/

We add neccessary decision procedures for our cache policy.


"""
# from opacus import PrivacyEngine


import numpy as np
import pandas as pd
# import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from agents.CacheAgent import LearnerAgent, RandomAgent, LRUAgent, LFUAgent


np.random.seed(1)

# Deep Q Network


class DQNAgent(LearnerAgent):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,

        e_greedy_min=(0.1, 0.1),
        e_greedy_max=(0.1, 0.1),

        # leave either e_greedy_init or e_greedy_decrement None to disable epsilon greedy
        # only leave e_greedy_increment to disable dynamic bidirectional epsilon greedy
        e_greedy_init=None,
        e_greedy_increment=None,
        e_greedy_decrement=None,

        reward_threshold=None,
        history_size=10,
        dynamic_e_greedy_iter=5,
        explore_mentor='LRU',

        replace_target_iter=300,
        memory_size=500,
        batch_size=32,

        output_graph=False,
        privacy=False,
        verbose=0
    ):
        self.ty = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.privacy = privacy

        self.epsilons_min = e_greedy_min
        self.epsilons_max = e_greedy_max
        self.epsilons_increment = e_greedy_increment
        self.epsilons_decrement = e_greedy_decrement

        self.epsilons = list(e_greedy_init)
        if (e_greedy_init is None) or (e_greedy_decrement is None):
            self.epsilons = list(self.epsilons_min)

        self.explore_mentor = None
        if explore_mentor.upper() == 'LRU':
            self.explore_mentor = LRUAgent
        elif explore_mentor.upper() == 'LFU':
            self.explore_mentor = LFUAgent

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

        # initialize a history set for rewards
        self.reward_history = []
        self.history_size = history_size
        self.dynamic_e_greedy_iter = dynamic_e_greedy_iter
        self.reward_threshold = reward_threshold

        # consist of [target_net, evaluate_net]
        # DNQ = DQNNet()
        self.Enet = DQNNet(self.n_features, self.n_actions).float()
        self.Tnet = DQNNet(self.n_features, self.n_actions).float()
        self.Tnet.load_state_dict(self.Enet.state_dict())
        self.Tnet.eval()

        self.optimizer = torch.optim.RMSprop(
            self.Enet.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.cost_his = []
        self.verbose = verbose

    def sync_Q_target(self):
        # self.net.target.load_state_dict(self.net.eval.state_dict())
        # print(self.Enet.state_dict())
        self.Tnet.load_state_dict(self.Enet.state_dict())

    def store_transition(self, s, a, r, s_):
        s, s_ = s['features'], s_['features']
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

        # Record reward
        if len(self.reward_history) == self.history_size:
            self.reward_history.pop(0)
        self.reward_history.append(r)

    def save_memory(self, dir):
        np.save(dir, self.memory)

    def load_memory(self, dir):

        self.memory_counter = self.memory_size+1
        self.memory = np.load(dir)

    def save_model(self, dir):
        torch.save(self.Enet, dir)

    def load_model(self, dir):
        torch.load(dir)

    def DP_model(self, sigma):
        nn.utils.clip_grad_norm_(self.Enet.parameters(), 1.0)
        for p in self.Enet.parameters():
            p.data.add_(sigma * torch.randn(p.size()))
        self.sync_Q_target()

    def choose_action(self, observation):
        # draw probability sample
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        elif self.epsilons[0] <= coin and coin < self.epsilons[0] + self.epsilons[1]:
            action = self.explore_mentor._choose_action(observation)
        else:
            observation = observation['features']
            # to have batch dimension when feed into tf placeholder
            state = torch.tensor(observation, dtype=torch.float)
            state = state.unsqueeze(0)
            action_values = self.Enet(state)
            action = torch.argmax(action_values, axis=1).item()

        if action < 0 or action > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % action)

        return action

    def choose_action_t(self, observation):
        # draw probability sample
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        else:

            state = torch.tensor(observation, dtype=torch.float)
            state = state.unsqueeze(0)
            action_values = self.Enet(state)
            action = torch.argmax(action_values, axis=1).item()

        if action < 0 or action > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % action)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # self.sess.run(self.replace_target_op)
            self.sync_Q_target()
            # verbose
            if self.verbose >= 1:
                print('Target DQN params replaced')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        state = torch.tensor(
            batch_memory[:, :self.n_features], dtype=torch.float)
        next_state = torch.tensor(
            batch_memory[:, -self.n_features:], dtype=torch.float)
        reward = torch.tensor(
            batch_memory[:, self.n_features + 1], dtype=torch.float)
        action = torch.tensor(
            batch_memory[:, self.n_features], dtype=torch.int64)
        # print("state", state, "next", next_state,
        #       "reward", reward, "actiomm", action)
        # q_next, q_eval = self.sess.run(
        #     [self.q_next, self.q_eval],
        #     feed_dict={
        #         self.s_: batch_memory[:, -self.n_features:],  # fixed params
        #         self.s: batch_memory[:, :self.n_features],  # newest params
        #     })
        # print(state[0])
        q_eval = self.Enet(state)[
            np.arange(0, self.batch_size), action
        ]
        # q = self.Enet(state)[
        #     np.arange(0, self.batch_size)
        # ]
        # print(torch.cat([state, q], dim=1))
        # change q_target w.r.t q_eval's action
        # q_target = q_eval.copy()

        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)

        next_state_Q = self.Enet(
            next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.Tnet(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        # print(reward)
        q_target = reward + self.gamma * next_Q
        # np.max(next_Q.detach().numpy(), axis=1)
        # q_target=(reward + (1 - done.float()) * self.gamma * next_Q).float()

        # q_target[batch_index, eval_act_index] = reward + \
        #     self.gamma * np.max(next_Q, axis=1)
        # train eval network
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # _, self.cost = self.sess.run([self._train_op, self.loss],
        #                              feed_dict={
        #                                  self.s: batch_memory[:, :self.n_features], self.q_target: q_target}
        #                              )
        self.cost_his.append(loss.item())
        # verbose
        if (self.verbose == 2 and self.learn_step_counter % 100 == 0) or \
                (self.verbose >= 3 and self.learn_step_counter % 20 == 0):
            print("Step=%d: Cost=%d" % (self.learn_step_counter, self.cost))

        # increasing or decreasing epsilons
        if self.learn_step_counter % self.dynamic_e_greedy_iter == 0:

            # if we have e-greedy?
            if self.epsilons_decrement is not None:
                # dynamic bidirectional e-greedy
                if self.epsilons_increment is not None:
                    rho = np.median(np.array(self.reward_history))
                    if rho >= self.reward_threshold:
                        self.epsilons[0] -= self.epsilons_decrement[0]
                        self.epsilons[1] -= self.epsilons_decrement[1]
                        # verbose
                        if self.verbose >= 3:
                            print("Eps down: rho=%f, e1=%d, e2=%f" %
                                  (rho, self.epsilons[0], self.epsilons[1]))
                    else:
                        self.epsilons[0] += self.epsilons_increment[0]
                        self.epsilons[1] += self.epsilons_increment[1]
                        # verbose
                        if self.verbose >= 3:
                            print("Eps up: rho=%f, e1=%d, e2=%f" %
                                  (rho, self.epsilons[0], self.epsilons[1]))
                # traditional e-greedy
                else:
                    self.epsilons[0] -= self.epsilons_decrement[0]
                    self.epsilons[1] -= self.epsilons_decrement[1]

            # enforce upper bound and lower bound
            def truncate(x, lower, upper): return min(max(x, lower), upper)
            self.epsilons[0] = truncate(
                self.epsilons[0], self.epsilons_min[0], self.epsilons_max[0])
            self.epsilons[1] = truncate(
                self.epsilons[1], self.epsilons_min[1], self.epsilons_max[1])

        self.learn_step_counter += 1


class DQNNet(nn.Module):

    def __init__(self, n_features, n_actions):
        super(DQNNet, self).__init__()

        self.online = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, input):
        x = self.online(input)
        return x
