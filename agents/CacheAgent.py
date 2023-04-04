import sys
import os
import random
import numpy as np
import pandas as pd

# Abstract class
# from agents.ReflexAgent import RandomAgent, LRUAgent, LFUAgent


class CacheAgent(object):
    def __init__(self, n_actions): pass
    def choose_action(self, observation): pass
    def store_transition(self, s, a, r, s_): pass


class ReflexAgent(CacheAgent):
    def __init__(self, n_actions): pass

    @staticmethod
    def _choose_action(n_actions): pass


class LearnerAgent(CacheAgent):
    def __init__(self, n_actions): pass
    def learn(self): pass


class RandomAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(n_actions):
        return random.randint(0, n_actions - 1)

    def choose_action(self, observation):
        return RandomAgent._choose_action(self.n_actions)


class LRUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        used_times = np.array(observation['last_used_times'])
        min_idx = np.argmin(used_times)
        return min_idx

    def choose_action(self, observation):
        min_idx = LRUAgent._choose_action(observation)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LRUAgent: Error index %d" % min_idx)
        return min_idx


class MRUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        used_times = np.array(observation['last_used_times'])
        max_idx = np.argmax(used_times)
        return max_idx

    def choose_action(self, observation):
        max_idx = MRUAgent._choose_action(observation)
        if max_idx < 0 or max_idx > self.n_actions:
            raise ValueError("MRUAgent: Error index %d" % max_idx)
        return max_idx


class LFUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        freq = observation['total_use_frequency']
        min_idx = np.argmin(freq)
        return min_idx

    def choose_action(self, observation):
        min_idx = LFUAgent._choose_action(observation)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LFUAgent: Error index %d" % min_idx)
        return min_idx


class FIFOAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        freq = observation['cached_times']
        min_idx = np.argmin(freq)
        return min_idx

    def choose_action(self, observation):
        min_idx = FIFOAgent._choose_action(observation)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LFUAgent: Error index %d" % min_idx)
        return min_idx


def simulate_network_latency(mean, std_dev):
    # 生成符合正态分布的随机数
    random_latency = random.normalvariate(mean, std_dev)
    # 将随机数转换为时延
    latency = max(0, random_latency)
    # 模拟网络拥塞，随机增加时延
    congestion = random.uniform(0, 1)
    latency *= (1 + congestion)
    # 返回时延
    return round(latency)

# 请求节点


class packets(object):
    def __init__(self, source, time, name, target, distance):
        self.source = source  # 源地址
        self.time = time  # 发出时间
        self.name = name  # 请求内容名字
        self.state = 1  # 包状态
        self.next_time = time + \
            simulate_network_latency(distance, distance/10)  # 下次处理时间
        self.last = source  # 上一节点
        self.target = target  # 下一节点
        self.history = []  # 历史记录

        self.iscopy = 0  # 0表示原生请求包，1表示是多播复制包
        self.copyfrom = ""  # 复制源头
        self.backcount = 0
        self.copyname = []

    def get_packets_info(self):
        return [self.source, self.time, self.name, self.state, self.next_time, self.last, self.target]

    def get_packets_last(self):
        return self.last

    def change_packets_state(self, time, state, distance, target):
        # 存入历史记录
        # print(self.source, self.time, self.name, self.state,
        #       self.next_time, self.last, self.target)
        self.history.append([time, self.state, self.target])
        self.last = self.target
        self.state = state
        self.next_time = time+simulate_network_latency(distance, distance/10)
        self.target = target

    def foward_packets(self, time, co_router_list, routing_table, pa):
        # 复制包并多播给其他协作路由节点
        forwardlist = []
        for node in co_router_list:
            if node != self.target:
                temp = packets(self.source, self.time, self.name,
                               self.target, 10)
                temp.state = 2
                temp.next_time = time + \
                    simulate_network_latency(
                        routing_table[self.target][node], routing_table[self.target][node]/10)
                temp.last = self.target  # 上一节点
                temp.target = node  # 下一节点
                temp.iscopy = 1  # 0表示原生请求包，1表示是多播复制包
                temp.copyfrom = pa  # 复制源头
                forwardlist.append(temp)
                self.backcount += 1
        # 原请求暂缓放置
        self.history.append([time, self.state, self.target])
        self.last = self.target
        self.state = 15
        self.next_time = -1
        return forwardlist

        # self.history.append([time, self.state, self.target])
        # self.last = self.target
        # self.state = state
        # self.next_time = time+simulate_network_latency(distance, distance/10)
        # self.target = target


class RequsetAgent(object):
    def __init__(self, id, pdf, v, num_files, taget, distance):
        super(RequsetAgent, self).__init__()
        self.id = id
        self.pdf = pdf
        self.v = v
        self.files = np.arange(num_files)
        self.target = taget
        self.distance = distance[id][taget]

    def get_packets(self, time):
        # 每一轮刚开始时生成请求
        if (time % self.v == 0):
            request = np.random.choice(self.files, size=1, p=self.pdf)
            packet = packets(self.id, time, request[0],
                             self.target, self.distance)
            return packet
        else:
            return 0

    def pdf_update(self, pdf):
        self.pdf = pdf

    def get_contents(self, time, packet):
        # 处理返回的请求包
        packet.change_packets_state(time, 16, 0, -1)
        return 16
    # def


# 路由节点
class RouterAgent(object):
    def __init__(self, id,  distance, cache_size, co, provider_id, agent):

        self.id = id
        self.distance = distance
        self.co = co
        self.provider_id = provider_id

        # 缓存初始化
        self.cache_size = cache_size
        self.FEAT_TREMS = [10, 100, 1000]
        self.slots = [-1] * self.cache_size  # 缓存插槽
        self.used_times = [-1] * self.cache_size  # 缓存上一次访问时间
        self.cached_times = [-1] * self.cache_size  # 缓存时间
        self.term_times = []  # 记录过去请求，用于统计频率

        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size

        self.resource_freq = {}
        self.slot_ids = 0  # 用于在填满缓存之前计数

        # 缓存决策
        self.agent = agent
        self.explore_mentor = LRUAgent
        self.cu_request = -1
        self.observation_ = []
        self.action_ = 0
        self.reward_params = dict(
            name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
        self.last_index = -1
        self.in_resource = -1
        self.out_resource = -1
        self.skip_resource = -1

    def get_contents(self, time, packet):
        # 路由节点根据传入的包作出反应
        info = packet.get_packets_info().copy()
        if info[3] == 1:  # 原生请求
            if info[2] not in self.resource_freq:
                self.resource_freq[info[2]] = 0
            self.resource_freq[info[2]] += 1

            self.term_times.append(info[2])

            if info[2] not in self.slots:
                if self.co == 1:  # 协作缓存，转发给其他路由节点
                    return 2

                else:  # 不协作，转发给远程节点
                    packet.change_packets_state(
                        time, 3, self.distance[self.id][self.provider_id], self.provider_id)
                    return 3
            else:  # 在本地缓存内找到，发回请求节点
                i = self.slots.index(info[2])
                self.used_times[i] = time

                packet.change_packets_state(
                    time, 4, self.distance[self.id][info[0]], info[0])
                return 4
        elif info[3] == 2:  # 协作请求

            if info[2] not in self.slots:
                packet.change_packets_state(
                    time, 6, self.distance[self.id][info[5]], info[5])

                return 6
            else:  # 在本地缓存内找到，发回请求节点
                i = self.slots.index(info[2])
                self.used_times[i] = time
                packet.change_packets_state(
                    time, 5, self.distance[self.id][info[5]], info[5])
                return 5
        # elif info[3] == 5:  # 协作返回
        #     packet.change_packets_state(
        #         time, 4, self.distance[self.id][info[0]], info[0])
        #     return 4
        # elif info[3] == 6:  # 协作失败
        #     return
        elif info[3] == 7:  # 远程返回
            packet.change_packets_state(
                time, 4, self.distance[self.id][info[0]], info[0])
            # 执行缓存替换和放置
            if self.slots.count(-1) > 0:  # 缓存未满
                i = self.slots.index(-1)
                self.slots[i] = info[2]
                self.used_times[i] = time
                self.cached_times[i] = time
            else:  # 缓存决策
                self.cu_request = info[2]
                self.cu_index = len(self.term_times)-1
                observation = self.get_observation()
                # 结算上一阶段奖励
                if isinstance(self.agent, LearnerAgent):
                    if self.observation_ != []:
                        reward = 0.0
                        hit_count = self.cu_index - self.last_index
                        reward += hit_count

                        miss_resource = self.cu_request
                        # If evction happens at last decision epoch
                        if self.action_ != 0:
                            # Compute the swap-in reward
                            past_requests = self.term_times[self.last_index:self.cu_index+1]
                            reward += self.reward_params['alpha'] * \
                                past_requests.count(self.in_resource)
                            # Compute the swap-out penalty
                            if miss_resource == self.out_resource:
                                reward -= self.reward_params['psi'] / \
                                    (hit_count + self.reward_params['mu'])
                        # Else no evction happens at last decision epoch
                        else:
                            # Compute the reward of skipping eviction
                            reward += self.reward_params['beta'] * reward
                            # Compute the penalty of skipping eviction
                            if miss_resource == self.skip_resource:
                                reward -= self.reward_params['psi'] / \
                                    (hit_count + self.reward_params['mu'])
                        self.agent.store_transition(
                            self.observation_, self.action_, reward, observation)

                action = self.agent.choose_action(observation)
                self.observation_ = observation
                self.action_ = action
                if action < 0 or action > len(self.slots):
                    raise ValueError("Invalid action %d taken." % action)

                self.last_index = len(self.term_times)-1
                # 缓存更换操作
                if action != 0:
                    self.out_resource = self.slots[action - 1]
                    self.in_resource = info[2]
                    slot_id = action - 1
                    self.slots[slot_id] = self.in_resource
                    self.cached_times[slot_id] = time
                    self.used_times[slot_id] = time
                    # self._hit_cache(slot_id)
                    # self.evict_count += 1
                else:
                    self.skip_resource = info[2]

            return 4
        else:
            print(info[3], "error")
        # if (time % self.v == 0):
        #     request = np.random.choice(self.files, size=1, p=self.pdf)
        #     packets = packets(time, request, self.target, self.distance)
        #     return packets
        # else:
        #     return 0

    def get_observation(self):
        return dict(features=self.get_features(),
                    cache_state=self.slots.copy(),
                    cached_times=self.cached_times.copy(),
                    last_used_times=self.used_times.copy(),
                    total_use_frequency=[
                        self.resource_freq.get(r, 0) for r in self.slots],
                    # access_bits=self.access_bits.copy(),
                    # dirty_bits=self.dirty_bits.copy()
                    )

    def get_features(self):
        # [Freq, F1, F2, ..., Fc] where Fi = [Rs, Rm, Rl]
        # i.e. the request times in short/middle/long term for each
        # cached resource and the currently requested resource.

        # base
        features = np.concatenate([
            np.array([self._elapsed_requests(t, self.cu_request) for t in self.FEAT_TREMS]), np.array(
                [self._elapsed_requests(t, rc) for rc in self.slots for t in self.FEAT_TREMS])
        ], axis=0)
        # last accessed time
        features = np.concatenate([
            features, np.array([self.used_times[i]
                                for i in range(self.cache_size)])
        ], axis=0)
        # cached time
        features = np.concatenate([
            features, np.array([self.cached_times[i]
                                for i in range(self.cache_size)])
        ], axis=0)
        return features

    # The number of requests on rc_id among last `term` requests.
    def _elapsed_requests(self, term, rc_id):
        # print(type(self.cur_index), type(term), term)
        l = len(self.term_times)
        start = l - term
        if start < 0:
            start = 0
        end = l
        # if end > len(self.requests):
        #     end = len(self.requests)
        return self.term_times[start: end].count(rc_id)


# 内容提供者节点
class ProviderAgent(object):
    def __init__(self, id,  distance):
        self.id = id
        self.distance = distance

    def get_contents(self, time, packet):
        # 包由远程请求状态改变为远程返回状态，发回给上一节点
        target = packet.get_packets_last()
        distance = self.distance[self.id][target]
        packet.change_packets_state(time, 7, distance, target)
        return 7
