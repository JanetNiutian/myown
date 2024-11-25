import pickle
import torch

# 设置矩阵乘法的精度为 'medium' 或 'high'
torch.set_float32_matmul_precision('medium')  # 或者根据需要选择 'high'

import numpy as np
import time
import copy
import random
import math
import torch.nn.functional as F

from get_data_for_val import get_data
# scene7
# from get_action_frenet_7 import *
# from get_action_frenet_11 import *
# from get_action_frenet_4 import *
from get_action_frenet_6 import *
from cost_func import cost_all

# 初始状态定义（包括自车和他车）
# 只是历史的最后一个时刻的状态
def state():
    startEgo_position=np.array(data['agent']['position'][-1,49,:2])
    startEgo_velocity=np.array(data['agent']['velocity'][-1,49,:2])
    startEgo_heading=np.array(data['agent']['heading'][-1,49])
    startEgo=np.hstack((startEgo_position,startEgo_velocity,startEgo_heading))
    
    state = copy.deepcopy(startEgo)
    
    for n in range(nobjs-1):
        position=np.array(data['agent']['position'][n,49,:2])
        velocity=np.array(data['agent']['velocity'][n,49,:2])
        heading=np.array(data['agent']['heading'][n,49])
        obj=np.hstack((position,velocity,heading))
        state = np.append(state, obj)
    
    return state

# 行为的定义
actions=list()
# actions.append(np.array(0.5))
actions.append(np.array(1.5))
# actions.append(np.array(2.5))
actions.append(np.array(3.5))
# actions.append(np.array(4.5))
actions.append(np.array(5.5))
# actions.append(np.array(6.5))
actions.append(np.array(7.5))
# actions.append(np.array(8.5))
actions.append(np.array(9.5))
# actions.append(np.array(10))
actions.append(np.array(11.5))
# actions.append(np.array(12.5))
actions.append(np.array(13.5))
actions.append(np.array(14.5))

# 处理 actions 列表中的每个元素
processed_actions = [a.reshape(1) if isinstance(a, np.ndarray) and a.ndim == 0 else a for a in actions]
actions = processed_actions

def _selectAction(s, h_s, d, iters):
    s=tuple(s)
    for _ in range(iters):
        _simulate(s, h_s, d)
    index = np.argmax([Q[(s,tuple(a))] for a in actions])
    a = actions[index]
    q = Q[(s,tuple(a))] # argmax
    return a

# tMaxRollouts设置是否应该和层深度以及6有关
def _simulate(s, h_s, d): 
    if d == 0: # we stop exploring the tree, just estimate Qval here
        return _rollout(s, h_s, d)  #返回的是q
    s=tuple(s)
    if s not in Tree:
        for a in actions:
            Nsa[(s,tuple(a))], Ns[s], Q[(s,tuple(a))] =  0, 1, 0. # could use expert knowledge as well
        Tree.add(s)
        # use tMax instead of d: we want to rollout deeper
        return _rollout(s, h_s,d)

    #a = max([(self.Q[(s,a)]+self.c*math.sqrt(math.log(self.Ns[s])/(1e-5 + self.Nsa[(s,a)])), a) for a in self.mdp.actions(s)])[1] # argmax
    qa_tab = ([(Q[(s,tuple(a))]+c*math.sqrt(math.log(Ns[s])/(1e-5 + Nsa[(s,tuple(a))])), a) for a in actions]) # argmax
    index = np.argmax([t[0] for t in qa_tab])
    qbest, a =  qa_tab[index]
    index = np.argmin([t[0] for t in qa_tab])
    qworst, _ = qa_tab[index]
    if abs(qbest - qworst) <= .1 or qbest <= -.55: # use exploration constant 1
        #pdb.set_trace()
        a = max([(Q[(s,tuple(a))] + 0.35 * math.sqrt(math.log(Ns[s])/(1e-5 + Nsa[(s,tuple(a))])), tuple(a)) for a in actions])# argmax
        a=a[1]
    
    # if d<=3:
    #     print("进行simulate，此时simulate深度为：",d)
    #     print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    sp, h_sp, r = _step(s, h_s, a)
    q = r + discount * _simulate(sp,h_sp, d-1)
    Nsa[(s,tuple(a))] += 1
    Ns[s] += 1
    Q[(s,tuple(a))] += (q-Q[(s,tuple(a))])/Nsa[(s,tuple(a))]
    return q

# 
def _rollout(s, h_s, d):
    if d == 0:
        return 0
    else:
        a = (random.sample(actions, 1))[0]
        # print("进行rollout，此时rollout深度为：",d)
        sp, h_sp, r = _step(s, h_s, a)
        return r + discount * _rollout(sp, h_sp, d-1)

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha  # 滤波系数，介于0和1之间，值越小滤波效果越强
        self.last_val = None  # 上一次的滤波结果

    def filter(self, val):
        if self.last_val is None:
            result = val
        else:
            result = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = result
        return result
    
def compute_heading(vx, vy):
    # 实例化滤波器，alpha根据需要调整
    vx_filter = LowPassFilter(alpha=0.5)
    vy_filter = LowPassFilter(alpha=0.5)

    # 更新速度值并获取滤波后的速度
    vx_filtered = vx_filter.filter(vx)  # vx是当前的速度x分量
    vy_filtered = vy_filter.filter(vy)  # vy是当前的速度y分量
    
    # 使用滤波后的速度计算航向角
    heading = math.atan2(vy_filtered, vx_filtered)
    return heading


def _step(state, history_state, action):
    sp = np.zeros_like(state)
    h_sp = copy.deepcopy(history_state)
    forecasted_trajectory=np.empty((10,nobjs,5))
    for i in range(nobjs-1):

        x = np.array(history_state['agent']['position'][i,49,0])
        y = np.array(history_state['agent']['position'][i,49,1])
        vx=np.array(history_state['agent']['velocity'][i,49,0])
        vy=np.array(history_state['agent']['velocity'][i,49,1])
        curvature=np.array(history_state['agent']['heading'][i,49])
        delta_t = 0.1

        forecasted_trajectory[0,i,0] = x+vx * delta_t
        forecasted_trajectory[0,i,1] = y+vy * delta_t
        forecasted_trajectory[0,i,2] = vx 
        forecasted_trajectory[0,i,3] = vy           
        forecasted_trajectory[0,i,4] = curvature

        # 逐步计算未来的状态
        for j in range(1,10):

            forecasted_trajectory[j,i,0] = forecasted_trajectory[j-1,i,0]
            forecasted_trajectory[j,i,1] = forecasted_trajectory[j-1,i,1]
            forecasted_trajectory[j,i,2] = vx 
            forecasted_trajectory[j,i,3] = vy           
            forecasted_trajectory[j,i,4] = curvature
    s = state[0:5]
    a = action
    # heading_all = np.empty((110,nobjs-1))
    heading_all = np.empty((10,nobjs-1))
    c_speed, c_accel, c_d, c_d_d, c_d_dd, s0 = initialize_frenet(s, history_state)
    x_ego, y_ego, heading_ego, vx_ego, vy_ego, s0_new, c_d_new, c_d_d_new, c_d_dd_new, c_speed_new, c_accel_new = get_ego_traj(s0, c_d, c_d_d, c_d_dd, c_speed, c_accel,a)
    forecasted_trajectory[:,-1,0] = x_ego[0]
    forecasted_trajectory[:,-1,1] = y_ego[0]
    forecasted_trajectory[:,-1,2] = vx_ego
    forecasted_trajectory[:,-1,3] = vy_ego
    forecasted_trajectory[:,-1,4] = heading_ego[0]
    sp[0] = x_ego[0][-1]
    sp[1] = y_ego[0][-1]
    sp[2] = vx_ego[-1]
    sp[3] = vy_ego[-1]
    sp[4] = heading_ego[0][-1]
    h_sp['agent']['position'][-1,:40,:]=history_state['agent']['position'][-1,10:50,:]
    h_sp['agent']['position'][-1,40:50,0]=torch.tensor(x_ego)
    h_sp['agent']['position'][-1,40:50,1]=torch.tensor(y_ego)
    h_sp['agent']['position'][-1,50:100,:]=history_state['agent']['position'][-1,60:,:]
    h_sp['agent']['position'][-1,100:,:]=torch.from_numpy(np.tile(history_state['agent']['position'][-1,109,:], (10, 1)))
    # h_sp['agent']['position'][-1,100:,:]=np.repeat(history_state['agent']['position'][-1,109,:], 10, axis=0)

    h_sp['agent']['heading'][-1,:40]=history_state['agent']['heading'][-1,10:50]
    h_sp['agent']['heading'][-1,40:50]=torch.tensor(heading_ego)
    h_sp['agent']['heading'][-1,50:100]=history_state['agent']['heading'][-1,60:]
    h_sp['agent']['heading'][-1,100:]=torch.from_numpy(np.tile(history_state['agent']['heading'][-1,109], 10))
    # np.repeat(history_state['agent']['heading'][-1,109], 10, axis=1)
    
    h_sp['agent']['velocity'][-1,:40,:]=history_state['agent']['velocity'][-1,10:50,:]
    h_sp['agent']['velocity'][-1,40:50,0]=torch.tensor(vx_ego)
    h_sp['agent']['velocity'][-1,40:50,1]=torch.tensor(vy_ego)
    h_sp['agent']['velocity'][-1,50:100,:]=history_state['agent']['velocity'][-1,60:,:]
    h_sp['agent']['velocity'][-1,100:,:]=torch.from_numpy(np.tile(history_state['agent']['velocity'][-1,109,:], (10, 1)))
    # np.repeat(history_state['agent']['velocity'][-1,109,:], 10, axis=1)

    # 用预测更新他车
    idx = 5

    for n in range(nobjs-1):

        h_sp['agent']['position'][n,:40,:] = history_state['agent']['position'][n,10:50,:]
        h_sp['agent']['position'][n,40:50,0] = torch.tensor(forecasted_trajectory[:,n,0])
        h_sp['agent']['position'][n,40:50,1] = torch.tensor(forecasted_trajectory[:,n,1])
        h_sp['agent']['position'][n,50:100,:] = history_state['agent']['position'][n,60:,:]
        h_sp['agent']['position'][n,100:,:] = history_state['agent']['position'][n,109,:].unsqueeze(0)


        h_sp['agent']['velocity'][n,:40,:] = history_state['agent']['velocity'][n,10:50,:]
        h_sp['agent']['velocity'][n,40:50,0] = torch.tensor(forecasted_trajectory[:,n,2])
        h_sp['agent']['velocity'][n,40:50,1] = torch.tensor(forecasted_trajectory[:,n,3])
        h_sp['agent']['velocity'][n,50:100,:] = history_state['agent']['velocity'][n,60:,:]
        h_sp['agent']['velocity'][n,100:,:] = history_state['agent']['velocity'][n,109,:].unsqueeze(0)

        h_sp['agent']['heading'][n,:40]= history_state['agent']['heading'][n,10:50]
        h_sp['agent']['heading'][n,40:50]= torch.tensor(forecasted_trajectory[:,n,4])
        h_sp['agent']['heading'][n,50:100]=history_state['agent']['heading'][n,60:]
        h_sp['agent']['heading'][n,100:]=history_state['agent']['heading'][n,109].unsqueeze(0)


        sp[idx] = forecasted_trajectory[-1,n,0]
        sp[idx+1] = forecasted_trajectory[-1,n,1]
        sp[idx+2] = forecasted_trajectory[-1,n,2]
        sp[idx+3] = forecasted_trajectory[-1,n,3]
        sp[idx+4] = forecasted_trajectory[-1,n,4]
        idx += 5

    return sp, h_sp, cost_all(forecasted_trajectory,nobjs)

# 需要选择并固定好这些参数在该场景下最合适的值，并选取好需要变动的值
# 从初始到结束的完整的循环
# 每一层采样都是best path，但是只选best path中的后0.1s作为下一层
# 每一层采样集合都是同一个centerline（或者人工规定某几层是同一个centerline后几层再变化），规定速度变化范围的集合，每层都从这相同的集合中采样

def full_rollout():
    s = state()
    s = tuple(s)  # just to make it hashable
    sp = np.zeros_like(s)
    h_s = data
    sequence = []
    c_speed, c_accel, c_d, c_d_d, c_d_dd, s0 = initialize_frenet(s, h_s)
    for steps in range(1, max_t+1):
        runtimes = []
        start_act = time.time()
        a = _selectAction(s, h_s, d, iters)
        end_act = time.time()
        runtimes.append(end_act - start_act)
        sequence.append(a)
        ego_state=s[0:5]
        print("第",steps,"次，选择动作",a)
        # print("第",steps,"次，自车状态",ego_state)
        # print("第",steps,"次，自车历史状态",h_s['agent']['position'][-1,47:52,:2])
      
        x_ego, y_ego, heading_ego, vx_ego, vy_ego, s0_new, c_d_new, c_d_d_new, c_d_dd_new, c_speed_new, c_accel_new = get_ego_traj(s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, a) # 不应该直接用_step,因为_step函数是转移1s
        # print(vx_ego)
        s0, c_d, c_d_d, c_d_dd, c_speed, c_accel = s0_new[1], c_d_new[1], c_d_d_new[1], c_d_dd_new[1], c_speed_new[1], c_accel_new[1]
        sp[0] = x_ego[0][0]
        sp[1] = y_ego[0][0]
        sp[2] = vx_ego[0]
        sp[3] = vy_ego[0]
        sp[4] = heading_ego[0][0]
        h_sp=copy.deepcopy(h_s)
        idx = 5
        for n in range(nobjs-1):
            position=np.array(h_s['agent']['position'][n,50,:2])
            velocity=np.array(h_s['agent']['velocity'][n,50,:2])
            heading=np.array(h_s['agent']['heading'][n,50])
            # obj=np.hstack((position,heading))
            # sp = np.append(sp, obj)
            sp[idx] = position[0]
            sp[idx+1] = position[1]
            sp[idx+2] = velocity[0]
            sp[idx+3] = velocity[1]
            sp[idx+4] = heading
            idx += 5

            h_sp['agent']['position'][n,:109,:]=h_s['agent']['position'][n,1:,:]
            h_sp['agent']['position'][n,109,:]=h_s['agent']['position'][n,109,:]
            h_sp['agent']['heading'][n,:109]=h_s['agent']['heading'][n,1:]
            h_sp['agent']['heading'][n,109]=h_s['agent']['heading'][n,109]
            h_sp['agent']['velocity'][n,:109,:]=h_s['agent']['velocity'][n,1:,:]
            h_sp['agent']['velocity'][n,109,:]=h_s['agent']['velocity'][n,109,:]
        
        h_sp['agent']['position'][-1,:49,:]=h_s['agent']['position'][-1,1:50,:]
        h_sp['agent']['position'][-1,49,0]=torch.tensor(sp[0])
        h_sp['agent']['position'][-1,49,1]=torch.tensor(sp[1])
        h_sp['agent']['position'][-1,50:109,:]=h_s['agent']['position'][-1,51:,:]
        h_sp['agent']['position'][-1,109,:]=h_s['agent']['position'][-1,109,:]

        h_sp['agent']['heading'][-1,:49]=h_s['agent']['heading'][-1,1:50]
        h_sp['agent']['heading'][-1,49]=torch.tensor(sp[4])
        h_sp['agent']['heading'][-1,50:109]=h_s['agent']['heading'][-1,51:]
        h_sp['agent']['heading'][-1,109]=h_s['agent']['heading'][-1,109]
    
        h_sp['agent']['velocity'][-1,:49,:]=h_s['agent']['velocity'][-1,1:50,:]
        h_sp['agent']['velocity'][-1,49,0]=torch.tensor(sp[2])
        h_sp['agent']['velocity'][-1,49,1]=torch.tensor(sp[3])
        h_sp['agent']['velocity'][-1,50:109,:]=h_s['agent']['velocity'][-1,51:,:]
        h_sp['agent']['velocity'][-1,109,:]=h_s['agent']['velocity'][-1,109,:]
        # print(sp[0:5])
        s = sp
        h_s=h_sp
        # runtime = np.mean(runtimes)
        metric_runtime.append(runtimes)
        ego_state_all.append(np.copy(sp[0:5]))
        # print(ego_state_all)
        chosed_actions.append(a)
        # print("完成第","次动作选择")
    return  metric_runtime,chosed_actions,ego_state_all


if __name__ == '__main__':
    # 输入的参数定义

    Q={}
    Tree=set()
    Nsa = {}
    Ns = {}
    # tMaxRollouts=10 #10 #200
    d=6 #30 #12
    iters=100 #50 #100
    max_t = 60 
    discount=1
    explorConst=1.0
    c=explorConst
    dt=0.1
    # dist_collision=10

    chosed_actions = []
    ego_state_all = []
    metric_runtime = []
    # cost_metric = []

    # scene7
    # raw_file_name = '3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24'
    # scene11
    # raw_file_name = '016b7b76-b5d0-4891-8205-3942113b05aa'
    # scene4
    # raw_file_name = '2ad5c87f-6e9f-4367-9f81-0589807411ad'
    # scene6
    raw_file_name = 'cf7c02bf-3d3e-4a10-a66a-f605d75a2f9e'
    data = get_data(raw_file_name)
    nobjs = data['agent']['num_nodes']

    metric_runtime,chosed_actions,ego_state_all = full_rollout()


  # 不是所有智能体都是定曲率预测

    print(metric_runtime,"\n",chosed_actions,"\n",ego_state_all)
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_runtime_6.pkl', 'wb')  #new
    pickle.dump(metric_runtime, f_save)
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_state_all_6.pkl', 'wb')  #new
    pickle.dump(ego_state_all, f_save)
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_actions_6.pkl', 'wb')  #new
    pickle.dump(chosed_actions, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_runtime_4.pkl', 'wb')  #new
    # pickle.dump(metric_runtime, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_state_all_4.pkl', 'wb')  #new
    # pickle.dump(ego_state_all, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_actions_4.pkl', 'wb')  #new
    # pickle.dump(chosed_actions, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_runtime_11.pkl', 'wb')  #new
    # pickle.dump(metric_runtime, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_state_all_11.pkl', 'wb')  #new
    # pickle.dump(ego_state_all, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_actions_11.pkl', 'wb')  #new
    # pickle.dump(chosed_actions, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_runtime_7.pkl', 'wb')  #new
    # pickle.dump(metric_runtime, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_state_all_7.pkl', 'wb')  #new
    # pickle.dump(ego_state_all, f_save)
    # f_save = open('/home/niutian/原data/QCNet_MCTS_result/cv_actions_7.pkl', 'wb')  #new
    # pickle.dump(chosed_actions, f_save)

