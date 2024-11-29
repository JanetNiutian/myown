"""
核心类:
    Node 类表示单个搜索树节点，负责存储动作及其概率、访问次数、评估值等。
    KrNode 是 Node 的包装器，提供更高级的接口。
Player 类:
    实现了 MCTS（蒙特卡洛树搜索）的核心逻辑，包括模拟、节点选择、扩展和评估。
    think 方法是主要入口，用于执行若干次模拟并返回最佳动作。
功能简化:  
    一些 Cython 特有的内容（如 cdef 和指针操作）被 Python 列表和对象管理所取代。
    predict 和 play_simulation 使用了占位逻辑，可以根据实际需求替换为真实实现。
"""

import time
from copy import copy
import numpy as np

from network.features import extract_planes
from config import Config
from utils import Utils


class Node:
    def __init__(self, x, y, spin, prob):
        self.x = x
        self.y = y
        self.spin = spin
        self.prob = prob
        self.children = []
        self.parent = None
        self.visits = 0
        self.eval = 0

    def add_node(self, x, y, spin, prob):
        new_node = Node(x, y, spin, prob)
        new_node.parent = self
        self.children.append(new_node)
        return new_node

    def sort_children(self, is_white):
        self.children.sort(
            key=lambda child: (child.visits, child.get_eval(is_white)), reverse=True
        )

    def ucb_select(self, is_white, ucb_const):
        return max(
            self.children,
            key=lambda child: child.get_eval(is_white) + ucb_const * (child.prob / (1 + child.visits)),
        )

    def get_eval(self, is_white):
        return self.eval if is_white else -self.eval

    def update_eval(self, value):
        self.visits += 1
        self.eval += value

    def is_root(self):
        return self.parent is None

    def has_children(self):
        return bool(self.children)


class KrNode:
    def __init__(self, x=0.0, y=0.0, spin=0.0, prob=0.0):
        # 初始化节点，存储动作和评估值
        self.x = x
        self.y = y
        self.spin = spin
        self.prob = prob
        self.parent = None
        self.children = []
        self.visits = 0
        self.eval_value = 0.0

    def add_node(self, x, y, spin, prob):
        # 添加子节点
        new_node = KrNode(x, y, spin, prob)
        new_node.parent = self
        self.children.append(new_node)
        return new_node

    def kr_update(self, value):
        # 更新节点的评估值
        self.eval_value = value

    def get_eval(self):
        # 获取节点的评估值
        return self.eval_value


class Player:
    def __init__(self, pipe, ucb_const, pw_const, init_temperature, out_temperature,
                 num_init, num_sample, l, max_depth, verbose):
        self.root_node = KrNode()
        self.pipe = pipe
        self.ucb_const = ucb_const
        self.pw_const = pw_const
        self.init_temperature = init_temperature
        self.out_temperature = out_temperature
        self.num_init = num_init
        self.num_sample = num_sample
        self.l = l
        self.max_depth = max_depth
        self.verbose = verbose
        self.reset(None, 0)

    def reset(self, root_env, num_playout):
        self.root_node = KrNode()
        self.root_env = root_env
        self.env_dict = {"": copy(self.root_env)}
        self.num_node = 0
        self.num_playout = num_playout

    def think(self, root_env, num_playout):
        self.reset(root_env, num_playout)
        root_prediction_p, root_leaf_black_eval = self.predict(root_env)
        self.prepare_init_actions(self.root_node, root_prediction_p)
        self.root_node.update_wval(root_leaf_black_eval) # 更新当前节点的评估值和访问次数

        for _ in range(num_playout):
            cur_env = copy(root_env)
            self.play_simulation(0, cur_env, self.root_node)

        return self.sample_best_action()

    def play_simulation(self, cur_depth, cur_env, cur_node):
        # 对应深度信息
        if cur_depth == self.max_depth:
            return None, False

        if cur_node.num_children < self.num_init:  # Expand
            init_action_id, init_action_prob = 0, 0.5  # Placeholder
            expanded_node = cur_node.add_node(0.0, 0.0, 0, init_action_prob)
            self.num_node += 1
            expanded_node.update_eval(0.5)  # Placeholder
            reward = self.calculate_reward(cur_env, (0.0, 0.0, 0))  # 奖励计算
            q = reward  # 因为这是叶子节点，直接返回奖励值
            expanded_node.update_eval(q)
            return q, True

        else:  # Selection
            selected_node = cur_node.ucb_select(cur_env.game_state["WhiteToMove"], self.ucb_const)
            selected_action = selected_node.action

            # 状态转移
            cur_env.step_without_rand(selected_action[0], selected_action[1], selected_action[2])

            # 递归调用
            reward = self.calculate_reward(cur_env, selected_action)
            leaf_black_eval, is_expanded = self.play_simulation(cur_depth + 1, cur_env, selected_node)
            q = reward + self.lambda_factor * leaf_black_eval  # 奖励累加

            selected_node.update_eval(q)  # 更新节点值
            return q, is_expanded
        
    # 通过预测的动作概率初始化动作
    # prediction_p是离散的，通过policy_network学习出来的
    def prepare_init_actions(self, node, prediction_p):
        # factor 用于调整每个初始化动作的概率。每选择一个新的初始化动作时，factor 会乘以一个系数，确保后续选择的动作概率递减，避免过度集中在少数几个动作上。
        # prev_init_action_prob用来存储上一个已选择的动作的概率值。在选择下一个初始化动作时，factor 会根据前一个动作的概率进行调整，以避免重复选择相同的动作，且动作选择概率逐步递减。
        factor = 1.0
        prev_init_action_prob = 0.0
        # 初始化动作个数由num_init决定
        for _ in range(self.num_init):
            # 从预测的概率分布中选择一个初始化动作
            # 这一行从 prediction_p 中选择一个动作索引，prediction_p 是一个包含所有可能动作的概率分布。self.apply_temperature()：是一个温度函数，用来调整概率分布的平滑程度，self.init_temperature 控制这个温度。温度调节越低，选择的动作越集中；温度越高，选择的动作越随机。
            init_action_id = np.random.choice(len(prediction_p), p=self.apply_temperature(prediction_p, self.init_temperature))
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

    def sample_best_action(self):
        children = self.root_node.children
        kns = np.array([child.visits for child in children])
        kns = kns / kns.sum()
        best_id = np.random.choice(np.arange(len(kns)), p=self.apply_temperature(kns, self.out_temperature))
        return children[best_id].action

    def apply_temperature(self, distribution, temperature):
        if temperature < 0.1:
            probabilities = np.zeros_like(distribution)
            probabilities[np.argmax(distribution)] = 1.0
        else:
            log_probabilities = np.log(distribution) / temperature
            probabilities = np.exp(log_probabilities - log_probabilities.max())
            probabilities /= probabilities.sum()
        return probabilities

    def predict(self, cur_env):
        if self.root_env.game_state["CurEnd"] != cur_env.game_state["CurEnd"]:
            # 如果游戏状态发生变化，则返回一个确定性的结果
            leaf_dist_v = [0] * Config.value.N_SCORE
            leaf_dist_v[Utils.value.score_to_idx(cur_env.game_state["Score"][self.root_env.game_state["CurEnd"]])] = 1.0
            leaf_black_eval = Utils.value.dist_v_to_exp_v(leaf_dist_v)
            return None, leaf_black_eval
        else:
            # 否则进行预测
            # 将当前环境转为特定输入形式
            input_planes = extract_planes(cur_env.game_state)  # 获取当前游戏状态的输入平面
            self.pipe.send(input_planes)  # 发送输入给神经网络
            prediction_p, prediction_v = self.pipe.recv()  # 从管道接收预测结果
            
            if cur_env.game_state["WhiteToMove"]:  # 如果是白方回合，翻转评估值
                prediction_v = Utils.value.flip_dist_v(prediction_v)
            
            # 将评估值转换为期望值
            leaf_black_eval = Utils.value.dist_v_to_exp_v(prediction_v)
            
            # 返回动作概率分布和评估值
            return prediction_p, leaf_black_eval
