
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
from cache.DataLoader import DataLoaderZipfpdf
from agents.CacheAgent import *
from agents.DQNAgent_p4 import DQNAgent


# 持续时间
timeduring = 10000
# 当前时间
current_time = 0
# 内容生成速率
v_files = 1
# 内容数
num_files = 10000
# 请求分布参数
parm = 1.3
# 协作参数,1协作，0不协作
co = 1
sw = 1

# 路由节点参数
cache_size = 50

# 生成请求的概率密度函数
requestpdf = DataLoaderZipfpdf(num_files, parm).get_pdf()
# print(requestpdf)

# 生成请求列表
requestlist = {}
recount = 0
# 待处理列表
usinglist = {}

# 处理完成列表
usedlist = {}
misscount = 0
backcount = 0

agent_num = 2
agent = []
userlist = []
router = []
nodelist = []
# agent1 = LFUAgent(cache_size+1)
# agent2 = LRUAgent(cache_size+1)
# agent = MRUAgent(cache_size+1)
# agent = FIFOAgent(cache_size+1)
for i in range(agent_num):
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

# 路由表
routing_table = [
    [0, -1, 10, -1, -1],
    [-1, 0, -1, 10, -1],
    [10, -1, 0, 20, 100],
    [-1, 10, 20, 0, 100],
    [-1, -1, 100, 100, 0]
]
# routing_table = [
#     [0,  10, -1],
#     [10, 0, 100],
#     [-1, 100, 0],
# ]
co_router_list = [2, 3]
provider_id = 4
#                        provider(5)
#                         /     \
#                        /       \
#                     100        100
#                      /           \
# user1(0)--10--router1(2)--20--router2(3)--10--user2(1)

# 节点定义
# 请求节点0,1
for i in range(agent_num):
    nodelist.append(0)
    nodelist.append(0)
for i in range(agent_num):
    userlist.append(RequsetAgent(i, requestpdf, v_files,
                                 num_files, i+agent_num, routing_table))
# 路由节点2,3
    nodelist[i] = userlist[i]
    router.append(RouterAgent(i+agent_num, routing_table,
                              cache_size, co, provider_id, agent[i]))
    nodelist[i+agent_num] = router[i]
# 远程节点4

p1 = ProviderAgent(4, routing_table)
nodelist.append(p1)

# user1 = RequsetAgent(0, requestpdf, v_files, num_files, 1, routing_table)
# # 路由节点2,3
# router1 = RouterAgent(1, routing_table, cache_size, co, provider_id, agent)
# # 远程节点4
# p1 = ProviderAgent(2, routing_table)
# nodelist = [user1,  router1,  p1]


for time in range(1, timeduring):
    # 生成请求,放入处理列表
    for user in userlist:
        temp = user.get_packets(time)
        if (temp != 0):
            requestlist[f"r{recount}"] = temp
            usinglist[f"r{recount}"] = temp
            recount += 1

    # 检查待处理列表中的包，查看是否有需要在当前处理的包
    for pa in usinglist.copy():
        info = usinglist[pa].get_packets_info().copy()
        if info[4] == time:
            if info[3] == 5:  # 协作返回，一个成功返回则不再处理其他转发包
                usinglist[usinglist[pa].copyfrom].change_packets_state(
                    time, 4, routing_table[info[5]][info[0]], info[0])
                for re in usinglist[usinglist[pa].copyfrom].copyname:
                    usinglist[re].change_packets_state(time, 16, 10, -1)
            elif info[3] == 6:  # 协作失败，继续等待，直到所有协作都返回失败后向远程请求
                # print(time, info, usinglist[usinglist[pa].copyfrom].backcount)
                if (usinglist[usinglist[pa].copyfrom].backcount > 1):
                    usinglist[pa].change_packets_state(time, 16, 10, -1)
                    usinglist[usinglist[pa].copyfrom].backcount = usinglist[usinglist[pa].copyfrom].backcount-1
                else:
                    usinglist[usinglist[pa].copyfrom].change_packets_state(
                        time, 3, routing_table[info[5]][provider_id], provider_id)
            else:
                # print(time, info)
                state = nodelist[info[6]].get_contents(time, usinglist[pa])
                if (state == 2):
                    # print(time, info)
                    # 执行协作转发，向协作节点复制请求并转发
                    forwardlist = usinglist[pa].foward_packets(
                        time, co_router_list, routing_table, pa)
                    for pac in forwardlist:
                        usinglist[f"r{recount}"] = pac
                        usinglist[pa].copyname.append(f"r{recount}")
                        recount += 1
                if (state == 7):  # 代表缓存未命中
                    misscount += 1
                if (state == 4):  # 代表缓存返回数
                    backcount += 1
        if info[3] == 16:
            # 清除无效包
            usedlist[pa] = usinglist[pa]
            usinglist.pop(pa)
    if (time > 1000) & (time % 1000 == 0):
        if isinstance(agent[0], LearnerAgent):
            if sw == 1:
                for i in range(agent_num):
                    os.makedirs('./data/agent'+str(i), exist_ok=True)
                    path = os.path.join('./data/agent'+str(i), 'agent.npy')
                    model_path = os.path.join(
                        './data/agent'+str(i), 'agent.pt')
                    agent[i].save_memory(path)
                    agent[i].save_model(model_path)
                    # agent[i].learn()
                input("savemodel")
                for i in range(agent_num):
                    path = os.path.join('./data/agent'+str(i), 'agent.npy')
                    model_path = os.path.join(
                        './data/agent'+str(i), 'agent.pt')
                    agent[i].load_memory(path)
                    agent[i].load_model(model_path)
            else:
                for i in range(agent_num):
                    agent[i].learn()
    if time % 1000 == 0:
        print(time, misscount/backcount)
        requestpdf = DataLoaderZipfpdf(num_files, parm).get_pdf()
        for user in userlist:
            user.pdf_update(requestpdf)

print(time, misscount/backcount)
