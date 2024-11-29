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
from val_for_MCTS import qc_for_state
from get_action_frenet_7 import *
from cost_func import cost_all
from full_route_7_visual import *

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

# 写论文时表明加速度和转向角的变化范围，实际操作优先尝试target speed的变化范围[0.5,15.5]
# AV2下最大限速为15m/s
# 研究出frenet在采样时为什么会阻止速度突然改变过多

# 为了存储每个动作的概率，需要定义节点
class KrNode(object): # 在 Python 3 中，所有的类默认都会隐式继承 object 类
    def __init__(self, speed=5.0,prob=0.25):    # 初始化时只存储动作信息
        self.speed = speed
        self.prob = prob
        self.init_actions = []  # 存储初始化动作和概率
        self.q_values = {}  # 存储 Q 值
        self.cov_matrix = np.identity(2)  # 默认协方差矩阵为单位矩阵
        self.actions = []  # 用于存储当前状态的所有动作集合 A
    def add_init_info(self, init_action_id, init_action_prob): # 将初始化动作及其概率添加到列表中
        self.init_actions.append((init_action_id, init_action_prob))
        self.q_values[init_action_id] = np.random.random()  # 给每个动作分配一个随机的Q值
        self.actions.append(init_action_id)  # 添加动作到 actions 集合中
    def get_init_info(self, nth): # 获取第 nth 个初始化动作
        return self.init_actions[nth]
    def get_q_value(self, action_id): # 获取给定动作的Q值
        return self.q_values.get(action_id, 0.0)  # 默认Q值为0.0,如果指定键的值不存在时，返回该默认值
    def update_q_value(self, action_id, new_q_value): # 更新给定动作的Q值
        self.q_values[action_id] = new_q_value
    def get_actions(self):
        """ 返回当前 KrNode 中存储的所有动作 """
        return self.actions
    def update_cov_matrix(self, actions, num_samples=5):
        """
        更新协方差矩阵。
        :param actions: 动作的集合。
        :param num_samples: 用于计算协方差矩阵的样本数量。
        """
        # 计算动作对之间的协方差
        action_matrix = np.array([a for a in actions])  # 假设 actions 是一个 numpy 数组
        sample_mean = np.mean(action_matrix, axis=0)
        diff = action_matrix - sample_mean
        self.cov_matrix = np.cov(diff.T)  # 计算协方差矩阵
    def get_max_q_action(self):
        """ 找到 Q 值最大的动作 """
        if not self.q_values:
            return None  # 如果没有动作，返回 None
        # 获取 Q 值最大的动作和对应的 Q 值
        max_action_id = max(self.q_values, key=self.q_values.get)
        return max_action_id, self.q_values[max_action_id]


# 行为的定义
def predict_policy(num):
    # 否则进行预测
    # 在 [1, 9] 区间平均选择 4 个动作
    actions = np.linspace(1, 9, num)  # 生成 4 个动作 [1, 4, 7, 9]
    # 每个动作的概率均匀分布，总和为 1
    prediction_p = np.ones(num) / num  # [0.25, 0.25, 0.25, 0.25] 
    # 返回动作概率分布
    return prediction_p
# 定义动作区间
# min_action = 0.5
# max_action = 15
# # 从区间 [0.5, 15] 内采样一个连续的动作
# sampled_action = np.random.uniform(min_action, max_action)
# # 在区间多次采样形成一系列连续的动作
# actions = [np.random.uniform(min_action, max_action) for _ in range(5)] # 输出[12.013293546568603, 7.62749571011462, 10.509056091831618, 8.048580024330351, 2.8766035953179096]
def apply_temperature(self, distribution, temperature=0.5):
    if temperature < 0.1:
        probabilities = np.zeros_like(distribution)
        probabilities[np.argmax(distribution)] = 1.0
    else:
        log_probabilities = np.log(distribution) / temperature
        probabilities = np.exp(log_probabilities - log_probabilities.max())
        probabilities /= probabilities.sum()
    return probabilities
def prepare_init_actions(node, prediction_p,num_init=4):
    # factor 用于调整每个初始化动作的概率。每选择一个新的初始化动作时，factor 会乘以一个系数，确保后续选择的动作概率递减，避免过度集中在少数几个动作上。
    # prev_init_action_prob用来存储上一个已选择的动作的概率值。在选择下一个初始化动作时，factor 会根据前一个动作的概率进行调整，以避免重复选择相同的动作，且动作选择概率逐步递减。
    factor = 1.0
    prev_init_action_prob = 0.0
    # 初始化动作个数由num_init决定
    for _ in range(num_init):
        # 从预测的概率分布中选择一个初始化动作
        # 这一行从 prediction_p 中选择一个动作索引，prediction_p 是一个包含所有可能动作的概率分布。self.apply_temperature()：是一个温度函数，用来调整概率分布的平滑程度，self.init_temperature 控制这个温度。温度调节越低，选择的动作越集中；温度越高，选择的动作越随机。
        init_action_id = np.random.choice(len(prediction_p), p=apply_temperature(prediction_p))
        init_action_prob = prediction_p[init_action_id]
        # 更新因子，避免重复选择
        # 每选择一个动作，更新因子 factor，确保选择动作的概率逐渐降低，避免重复选择。prev_init_action_prob 是前一个动作的概率，1.0 - prev_init_action_prob 保证了后续动作的选择概率会随着选择的动作数量增加而递减。
        factor *= (1.0 - prev_init_action_prob)
        # 将当前选择的动作概率保存为 prev_init_action_prob，供下次循环时更新 factor 使用
        prev_init_action_prob = init_action_prob
        # 将初始化动作及其概率添加到节点
        node.add_init_info(init_action_id, init_action_prob * factor)
        # 将已选动作的概率设为 0，避免再次选择
        prediction_p[init_action_id] = 0.0
        prediction_p += 1.0E-6  # 添加一个小的噪声，防止概率为零
        prediction_p /= prediction_p.sum()  # 重新归一化概率分布

# 初始A的确定
# 处理 actions 列表中的每个元素
# processed_actions = [a.reshape(1) if isinstance(a, np.ndarray) and a.ndim == 0 else a for a in actions]
# actions = processed_actions
# 还需要给出简单的policy更新，先让policy network就是预测结果的反应
# 这里采用target speed的均匀分布

def _selectAction(s, h_s, d, iters):
    s=tuple(s)
    for _ in range(iters):
        _simulate(s, h_s, d)
    node = state_to_node[s]
    max_action, _ = node.get_max_q_action()
    return max_action

# 增加候选action的选取
def K(a, b, cov_matrix):
    """
    高斯核函数，计算动作 a 和 b 之间的相似度，使用协方差矩阵来加权。
    :param a: 动作 a 的特征。
    :param b: 动作 b 的特征。
    :param cov_matrix: 协方差矩阵。
    :return: 核函数值。
    """
    diff = a - b
    inv_cov = np.linalg.inv(cov_matrix)
    return np.exp(-0.5 * diff.T @ inv_cov @ diff) / (2 * np.pi * np.linalg.det(cov_matrix))**0.5

def W(a, actions, Nsa, cov_matrix):
    """
    计算动作 a 的权重 W(a)，基于高斯核函数。
    :param a: 当前动作。
    :param actions: 所有动作的集合。
    :param Nsa: (状态, 动作) 对应的访问次数。
    :param cov_matrix: 协方差矩阵。
    :return: 动作的权重。
    """
    weight = 0
    for b in actions:
        weight += K(a, b, cov_matrix) * Nsa.get((tuple(a), tuple(b)), 1)  # 默认 Nsa 值为1，避免除零
    return weight

def E_v_a(a, actions, Q, Nsa, cov_matrix):
    """
    计算给定动作 a 的期望价值。
    :param a: 当前动作。
    :param actions: 所有动作的集合。
    :param Q: Q 值字典。
    :param Nsa: (状态, 动作) 对应的访问次数。
    :param cov_matrix: 协方差矩阵。
    :return: 动作 a 的期望价值。
    """
    numerator = 0.0
    denominator = 0.0
    for b in actions:
        k_ab = K(a, b, cov_matrix)
        numerator += k_ab * Q.get((tuple(a), tuple(b)), 0) * Nsa.get((tuple(a), tuple(b)), 1)
        denominator += k_ab * Nsa.get((tuple(a), tuple(b)), 1)
    
    return numerator / (denominator + 1e-5)  # 防止除零错误

def _simulate(s, h_s, d): 
    if d == 0: # we stop exploring the tree, just estimate Qval here
        return _value(s, h_s, d)  #返回的是q
    s=tuple(s)
    # 检查状态是否已经存在于字典中
    if s not in state_to_node:
        # 如果状态不存在，创建一个新的 KrNode，并添加到字典中
        node = KrNode()
        # 假设我们有一个动作概率分布预测，num_init 为 4
        prediction_p = predict_policy(4)  # 假设预测为 [0.25, 0.25, 0.25, 0.25]
        prepare_init_actions(node, prediction_p, num_init=4)  # 初始化动作和它们的 Q 值
        # 将新的 KrNode 存入字典
        state_to_node[s] = node
        return _value(s, h_s,d)
    else:
        # 如果状态已经存在，直接获取对应的 KrNode
        print("why???")
        node = state_to_node[s]
    # for i in range(len(node.init_actions)):
    #     action_id, prob = node.get_init_info(i) # 获得初始化的动作和概率
    #     q_value = node.get_q_value(action_id)

    # Step 1: `a' ← argmin_{K(a,b) > γ} W(b)`
    best_action = None
    best_w = float('inf')  # 最小化 W(b)
    actions = node.get_actions()
    A = [a for a in actions]
    node.update_cov_matrix(A, num_samples=len(A))
    for a in actions:
        for b in A:
            if K(a, b, node.cov_matrix) > gamma:  # K(a, b) > γ
                w_b = W(b, actions, Nsa, node.cov_matrix)
                if w_b < best_w:  # 选择最小的 W(b)
                    best_w = w_b
                    best_action = a
    # Step 2: `A ← A ∪ a'`
    if best_action is not None:
        A.append(best_action)
    # Step 3: `a ← argmax_{a ∈ A} E[\bar{v}_a | a] + C * sqrt(log(∑_{b ∈ A} W(b)) / W(a))`
    best_value = float('-inf')
    selected_action = None
    A = [a for a in actions]
    node.update_cov_matrix(A, num_samples=len(A))
    total_w = sum(W(b, actions, Nsa, node.cov_matrix) for b in A)  # ∑_{b ∈ A} W(b)
    
    for a in A:
        q_value = E_v_a(a, actions, Q, Nsa, node.cov_matrix)  # 计算期望值
        exploration_term = c * math.sqrt(math.log(total_w) / (1e-5 + W(a, actions, Nsa, node.cov_matrix)))  # 防止除零
        value = q_value + exploration_term
        
        if value > best_value:
            best_value = value
            selected_action = a
    
    sp, h_sp, r = _step(s, h_s, selected_action)
    q = r + discount * _simulate(sp,h_sp, d-1)
    Nsa[(s,tuple(selected_action))] += 1
    Ns[s] += 1
    new_q_value = (q-node.get_q_value(selected_action))/Nsa[(s,tuple(selected_action))]
    node.update_q_value(selected_action, new_q_value)
    return q

# rollout替换为直接评估方法
# 简略成当前场景的cost和未来剩余时常的
# discount ^ (6-d) * cost
def _value(s, h_s, d):
    return cost_all(h_s,nobjs)*discount**(6-d) 

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

# step主要表示转移函数
# 按照这个速度进行frenet采样
# 采样之后获得轨迹，轨迹的0.1s作为下一个state
# 下一个state确定之后，计算这一个 时刻 的cost 作为reward
def _step(state, history_state, action):
    sp = np.zeros_like(state)
    h_sp = copy.deepcopy(history_state)
    forecasted_trajectory=np.empty((10,nobjs,5)) # 方便计算reward
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
    qc_output = qc_for_state(history_state)
    eval_mask=3
    pi=qc_output['pi']
    pi_eval = F.softmax(pi[eval_mask], dim=-1)
    pi_eval = pi_eval.cpu().numpy()
    random_index = np.random.choice([0, 1, 2, 3, 4, 5], p=pi_eval.ravel())
    for n in range(nobjs-1):
        s_obj = state[idx:idx+3]
        theta = history_state['agent']['heading'][n][49]

        # # 打印 theta 的值以进行调试
        # print("theta:", theta)
        # # 检查 theta 的范围，确保在 -π 到 π 之间
        # if not -math.pi <= theta <= math.pi:
        #     raise ValueError(f"theta 值超出范围: {theta}")
        
        cos = math.cos(theta)
        sin = math.sin(theta)
        origin = history_state['agent']['position'][n][49]
        loc_refine_pos = qc_output['loc_refine_pos'][n][random_index].cpu()

        # print(loc_refine_pos.shape) torch.Size([60, 2])
        # print((origin[0].unsqueeze(0)).shape) torch.Size([1])

        x = loc_refine_pos[:10,0]*cos-loc_refine_pos[:10,1]*sin+origin[0].unsqueeze(0) # loc_refine_pos[6,60,2]
        y = loc_refine_pos[:10,0]*sin+loc_refine_pos[:10,1]*cos+origin[1].unsqueeze(0) # [random_index,0,0]
        h_sp['agent']['position'][n,:40,:] = history_state['agent']['position'][n,10:50,:]
        h_sp['agent']['position'][n,40:50,0] = x
        h_sp['agent']['position'][n,40:50,1] = y
        h_sp['agent']['position'][n,50:100,:] = history_state['agent']['position'][n,60:,:]
        h_sp['agent']['position'][n,100:,:] = history_state['agent']['position'][n,109,:].unsqueeze(0)

        v_x = (h_sp['agent']['position'][n,40:50,0]-h_sp['agent']['position'][n,39:49,0])/0.1
        v_y = (h_sp['agent']['position'][n,40:50,1]-h_sp['agent']['position'][n,39:49,1])/0.1

        h_sp['agent']['velocity'][n,:40,:] = history_state['agent']['velocity'][n,10:50,:]
        h_sp['agent']['velocity'][n,40:50,0] = v_x
        h_sp['agent']['velocity'][n,40:50,1] = v_y
        h_sp['agent']['velocity'][n,50:100,:] = history_state['agent']['velocity'][n,60:,:]
        h_sp['agent']['velocity'][n,100:,:] = history_state['agent']['velocity'][n,109,:].unsqueeze(0)

        for hn in range(10):
            heading_all[hn,n] = compute_heading(v_x[hn], v_y[hn])

        # for hn in range(10):
        #     speed = math.sqrt(v_x[hn]**2 + v_y[hn]**2)
        #     min_speed_threshold=2.0
        #     if speed > min_speed_threshold:
        #         heading_all[50+hn,n] = compute_heading(v_x[hn], v_y[hn])
        #     else:
        #         heading_all[50+hn,n] = heading_all[50+hn-1,n]


        h_sp['agent']['heading'][n,:40]= history_state['agent']['heading'][n,10:50]
        h_sp['agent']['heading'][n,40:50]= torch.FloatTensor(heading_all[:,n])
        # qc_output['loc_refine_head'][n,random_index,:10,0].cpu()+theta.unsqueeze(0)   
        # torch.FloatTensor(heading)
        h_sp['agent']['heading'][n,50:100]=history_state['agent']['heading'][n,60:]
        h_sp['agent']['heading'][n,100:]=history_state['agent']['heading'][n,109].unsqueeze(0)

        forecasted_trajectory[:,n,0] = x
        forecasted_trajectory[:,n,1] = y
        forecasted_trajectory[:,n,2] = v_x
        forecasted_trajectory[:,n,3] = v_y
        forecasted_trajectory[:,n,4] = heading_all[:,n]

        sp[idx] = x[-1]
        sp[idx+1] = y[-1]
        sp[idx+2] = v_x[-1]
        sp[idx+3] = v_y[-1]
        sp[idx+4] = heading_all[-1,n]
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
    # 创建一个字典用于存储状态对应的 KrNode
    state_to_node = {}
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

    raw_file_name = '3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24'
    data = get_data(raw_file_name)
    nobjs = data['agent']['num_nodes']

    metric_runtime,chosed_actions,ego_state_all = full_rollout()

    # cost_metric的计算需要根据自车的ego_state_all和他车未来6s的真值，输出评价指标
    # Finish Time , 画出自车轨迹，设置目标点，通过ego_state_all看出到达目标点用了多少step
    # average velocity,  用ego_state_all就能计算

    # collision rate, 自车和其他车以及车道线小于车边长的


    print(metric_runtime,"\n",chosed_actions,"\n",ego_state_all)
    print(f"QCNet被调用了:{qc_for_state.count}次")
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/metric_runtime_7.pkl', 'wb')  #new
    pickle.dump(metric_runtime, f_save)
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/ego_state_all_7.pkl', 'wb')  #new
    pickle.dump(ego_state_all, f_save)
    f_save = open('/home/niutian/原data/QCNet_MCTS_result/chosed_actions_7.pkl', 'wb')  #new
    pickle.dump(chosed_actions, f_save)

    
    # f_read = open('/home/niutian/原data/QCNet_MCTS_result/ego_state_all_7.pkl', 'rb')
    # ego_state_all = pickle.load(f_read)
    visualization(ego_state_all,nobjs)

    # simulater每次深度最多到45，目前看没有太大问题
    # 把指标打印出来对比

    ###########
    # 先跑出来本身指标，再计算简单模型 # 选择不体现交互的
    
    ####### 对比 #######
    # 也是mcts但是默认他车更新方式是速度曲率不变
    # (也是mcts但是没有用frenet)
    # (mpc)


    # conda activate QCNet
    # cd /home/niutian/原data/code/QCNet-main-scene7
    # CUDA_VISIBLE_DEVICES=2 python3 /home/niutian/原data/code/QCNet-main-scene7/full_route_7.py
