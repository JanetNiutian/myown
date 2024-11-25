import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要GUI

import pickle
import math
from frenet_optimal_trajectory_scene7 import *
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file
from pathlib import Path

from get_data_for_val import get_data

scenario_path='/home/niutian/原data/Argoverse 2 Motion Forecasting Dataset/val/raw/3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24/scenario_3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24.parquet'
df = pd.read_parquet(scenario_path)
static_map_path=Path('/home/niutian/原data/Argoverse 2 Motion Forecasting Dataset/val/raw/3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24/log_map_archive_3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24.json')
map_data = read_json_file(static_map_path)

map_api = ArgoverseStaticMap.from_json(static_map_path)
raw_file_name = '3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24'
data = get_data(raw_file_name)
nobjs = data['agent']['num_nodes'] #26

x_start = np.array(data['agent']['position'][-1,49,0])
y_start = np.array(data['agent']['position'][-1,49,1])

v_x_start = data['agent']['velocity'][-1,49,0]
v_y_start = data['agent']['velocity'][-1,49,1]

heading_start = data['agent']['heading'][-1,49]

query_center = np.array([x_start, y_start])
search_radius_m = 1000
nearby_lane_segments = map_api.get_nearby_lane_segments(query_center, search_radius_m)
# all_ids = map_api.get_scenario_lane_segment_ids
number_lane = 0
x_middle = -39.2399
y_middle = 2325.2805
x_end = -83.68512
y_end = 2323.4067
# plt.scatter(x_start, y_start,color='black')
# plt.scatter(x_middle, y_middle,color='black')
# plt.scatter(x_end, y_end,color='black')
for LaneSegment in nearby_lane_segments: 
    number_lane += 1
    length = len(LaneSegment.polygon_boundary)
    x = np.zeros([length])
    y = np.zeros([length])
    for i in range(length):
        x[i] = LaneSegment.polygon_boundary[i,0]
        y[i] = LaneSegment.polygon_boundary[i,1]
    # if number_lane == 26 or number_lane == 45 or number_lane == 48 or number_lane == 62 or number_lane == 90 or number_lane == 95:   
    #     print(LaneSegment.id) 
        # plt.plot(x,y)
        # plt.axis('equal')
    #     plt.savefig('/home/niutian/原data/planner_result/scene7_lines%d.png'%number_lane)
    # plt.savefig('/home/niutian/原data/planner_result/scene7_test.png')

centerline_1 = map_api.get_lane_segment_centerline(438976671)
centerline_2 = map_api.get_lane_segment_centerline(438977465)
centerline_3 = map_api.get_lane_segment_centerline(438977401)
centerline_4 = map_api.get_lane_segment_centerline(438977061)
centerline_5 = map_api.get_lane_segment_centerline(438976542)
centerline_6 = map_api.get_lane_segment_centerline(438974309)
centerline_point = np.vstack([centerline_6, centerline_5, centerline_1, centerline_2, centerline_3, centerline_4])
# print(centerline_point)
centerline_x = centerline_point[:, 0]
centerline_y = centerline_point[:, 1]
# print(centerline_point)
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# 从参考点中提取 x 和 y 坐标
xy_points = centerline_point[:, :2]  # 假设高度（z坐标）不影响平面距离计算
distances = np.sqrt(np.sum(np.diff(xy_points, axis=0)**2, axis=1))

# 找到距离大于某个小阈值的点（这里假设为1e-8，但可以根据实际情况调整）
valid_indices = np.where(distances > 1e-8)[0] + 1

# 总是包含第一个点
valid_indices = np.insert(valid_indices, 0, 0)

def extended_fp(centerline):
    # 将中心线按第一维（x 轴）从大到小排序
    sorted_centerline = centerline[np.argsort(centerline[:, 0])[::-1]]

    # 计算原始中心线的总长度
    original_length = np.sum(np.linalg.norm(np.diff(sorted_centerline, axis=0), axis=1))

    # 获取排序后最后两个点计算方向向量
    direction_vector = sorted_centerline[-1] - sorted_centerline[-2]
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # 归一化方向向量

    # 生成新的点，直到延长长度达到原始长度
    new_points = []
    current_point = sorted_centerline[-1]
    extended_length = 0

    while extended_length < original_length*10:
        new_point = current_point + direction_vector
        new_points.append(new_point)
        extended_length += np.linalg.norm(direction_vector)
        current_point = new_point

    new_points = np.array(new_points)

    # 合并原始中心线点和新的延长点
    extended_centerline = np.vstack((sorted_centerline, new_points))
    return extended_centerline

# 使用有效索引过滤点
filtered_points = xy_points[valid_indices]
extended_filtered_points = extended_fp(filtered_points)

distances = np.sqrt(np.sum(np.diff(extended_filtered_points, axis=0)**2, axis=1))
# 找到距离大于某个小阈值的点（这里假设为1e-8，但可以根据实际情况调整）
valid_indices = np.where(distances > 1e-8)[0] + 1
# 总是包含第一个点
valid_indices = np.insert(valid_indices, 0, 0)
valid_extended = extended_filtered_points[valid_indices]
# f_save = open('/home/niutian/原data/code/QCNet-main/centerline_7.pkl', 'wb')  #new
# # 把filter points plt并延长
# pickle.dump(valid_extended, f_save)
# print(filtered_points)
# plt.plot(filtered_points[:, 0],filtered_points[:, 1])
# plt.axis('equal')
# plt.savefig('/home/niutian/原data/planner_result/scene7_filter_point.png')

# print("finish save centerline")
# 计算弧长s
s = np.zeros(valid_extended.shape[0])
s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(valid_extended, axis=0)**2, axis=1)))
cs_x = CubicSpline(s, valid_extended[:, 0])
cs_y = CubicSpline(s, valid_extended[:, 1])

# 计算给定点到曲线的最短距离
def distance_to_curve(s, x0, y0):
    x = cs_x(s)
    y = cs_y(s)
    return np.sqrt((x - x0)**2 + (y - y0)**2)

# 使用数值优化找到最近点的s值
result = minimize_scalar(distance_to_curve, args=(x_start, y_start), bounds=(s[0], s[-1]), method='bounded')

# 使用s值计算最近点的坐标、方向角和曲率
s_nearest = result.x
x_nearest = cs_x(s_nearest)
y_nearest = cs_y(s_nearest)
# 计算方向角 theta_r
dx_ds = cs_x.derivative()(s_nearest)
dy_ds = cs_y.derivative()(s_nearest)
theta_nearest = np.arctan2(dy_ds, dx_ds)
# 计算曲率 k_r，调整获取二阶导数的方法
dx_ds1 = cs_x.derivative()  # 一阶导数的样条对象
dy_ds1 = cs_y.derivative()  # 一阶导数的样条对象

d2x_ds2 = dx_ds1.derivative()(s_nearest)  # 二阶导数值
d2y_ds2 = dy_ds1.derivative()(s_nearest)  # 二阶导数值
k_nearest = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)
# 定义一个小的增量delta_s，用于计算数值导数
delta_s = 1e-4
s_plus = s_nearest + delta_s
s_minus = s_nearest - delta_s

# 计算 s_nearest, s_plus, 和 s_minus 处的曲率
def curvature(s):
    dx_ds = cs_x.derivative()(s)
    dy_ds = cs_y.derivative()(s)
    d2x_ds2 = cs_x.derivative().derivative()(s)
    d2y_ds2 = cs_y.derivative().derivative()(s)
    return (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)

k_r = curvature(s_nearest)
k_r_plus = curvature(s_plus)
k_r_minus = curvature(s_minus)

# 使用中心差分公式计算 k'_r
k_r_prime = (k_r_plus - k_r_minus) / (2 * delta_s)
from math import *
def cartesian2Frenet(x,y,theta_x,v_x,a_x,k_x,s_r,x_r,y_r,theta_r,k_r,k_r_):
    """全局坐标系转Frenet坐标系

    Args:
        x (_type_): Cartesian坐标系下的车辆横坐标位置
        y (_type_): Cartesian坐标系下的车辆纵坐标位置
        theta_x (_type_): 为方位角，即全局坐标系下的朝向；
        v_x (_type_): Cartesian坐标系下的线速度大小;
        a_x (_type_): Cartesian坐标系下的加速度
        k_x (_type_): 曲率
        s_r (_type_): 投影点的弧长
        x_r (_type_): 投影点P点在Cartesian坐标系下的x坐标
        y_r (_type_): 投影点P点在Cartesian坐标系下的y坐标
        theta_r (_type_): 投影点P点在Cartesian坐标系下的朝向角
        k_r (_type_): 曲率
        k_r_ (_type_): 曲率对弧长s的一阶导数_

    Returns:
        _type_: Frenet坐标系下车辆的运动状态
    """
    # 计算自车位置到参考线最近点的向量
    dx = x - x_r
    dy = y - y_r
    
    # 计算向量在参考线方向上的投影长度
    # 这里假设theta_r是参考线在最近点的方向角
    delta_x = np.cos(theta_r) * dx + np.sin(theta_r) * dy
    delta_y = -np.sin(theta_r) * dx + np.cos(theta_r) * dy
   
    # 横向距离d可以从delta_y获得
    d = delta_y  # 注意：d的符号表明自车相对参考线的左侧（正）或右侧（负）
    delta_theta = theta_x-theta_r
    one_kr_d = 1-k_r*d
    s=s_r
    d=np.sign((y-y_r)*cos(theta_r)-(x-x_r)*sin(theta_r))*sqrt((x-x_r)**2+(y-y_r)**2)
    dot_d = v_x*sin(delta_theta)
    ddot_d = a_x*sin(delta_theta)
    dot_s=v_x*cos(delta_theta)/one_kr_d
    d_=one_kr_d*tan(delta_theta)
    d__=-(k_r_*d+k_r*d_)*tan(delta_theta)+one_kr_d/(cos(delta_theta))**2*(k_x*one_kr_d/cos(delta_theta)-k_r)
    ddot_s = (a_x*cos(delta_theta)-dot_s**2*(d_*(k_x*one_kr_d/cos(delta_theta)-k_r)-(k_r_*d+k_r*d_)))/one_kr_d

    return s,dot_s,ddot_s,d,dot_d,ddot_d,d_,d__

 # 自车当前状态转化为 frenet 坐标系
x_p1 = data['agent']['position'][-1,48,0]
y_p1 = data['agent']['position'][-1,48,1]

v_x_p1 = data['agent']['velocity'][-1,48,0]
v_y_p1 = data['agent']['velocity'][-1,48,1]

heading_p1 = data['agent']['heading'][-1,48]

x_p2 = data['agent']['position'][-1,47,0]
y_p2 = data['agent']['position'][-1,47,1]

v_x_p2 = data['agent']['velocity'][-1,47,0]
v_y_p2 = data['agent']['velocity'][-1,47,1]

heading_p2 = data['agent']['heading'][-1,47]

def calculate_curvature(p1, p2, p3):
    # 将点转换为numpy数组
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # 计算向量
    vec1 = p2 - p1
    vec2 = p3 - p2
    
    # 计算向量的叉积和模长
    cross_product = np.cross(vec1, vec2)
    norm_cross_product = np.linalg.norm(cross_product)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    norm_vec3 = np.linalg.norm(p3 - p1)
    
    # 使用曲率公式进行计算
    curvature = 2 * norm_cross_product / (norm_vec1 * norm_vec2 * norm_vec3)
    
    return curvature

# 示例点
p1 = [x_p2, y_p2]
p2 = [x_p1, y_p1]
p3 = [x_start, y_start]
# 还需要获得曲率curvature k 和 加速度
# 计算曲率
curvature = calculate_curvature(p1, p2, p3)
dt=0.1
v2 = np.array([v_x_start, v_y_start]) 
v1 = np.array([v_x_p1, v_y_p1]) 
acceleration = (v2 - v1) / dt

from math import *

x = x_start
y = y_start
v = math.sqrt(v_x_start**2 + v_y_start**2)  # 1.78859228093272
a = math.sqrt(acceleration[0]**2 + acceleration[1]**2) # 0.5462798942874714
theta = heading_start
kappa = curvature

# s0,c_speed,c_accel,c_d,dot_d,ddot_d,c_d_d,c_d_dd = cartesian2Frenet(x,y,theta,v,a,kappa,s_nearest,x_nearest,y_nearest,theta_nearest,k_nearest,k_r_prime)

# csp = cubic_spline_planner.CubicSpline2D(filtered_points[:, 0], filtered_points[:, 1])
# fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
# fplist = calc_global_paths(fplist, csp)
# fplist = check_paths(fplist, ob)


# # 获得了笛卡尔坐标系的轨迹集合

# min_cost = float("inf")
# best_path = None
# for fp in fplist:
#     if min_cost >= fp.cf:
#         min_cost = fp.cf
#         best_path = fp
# number = 0
# for fp in fplist:  
#     number += 1         
#     plt.plot(fp.x,fp.y)


# plt.plot(filtered_points[:, 0], filtered_points[:, 1])
# plt.plot(best_path.x,best_path.y,color = 'black')


# plt.axis('equal')
# plt.savefig('/data/niutian/planner_result/scene7_test.png')
# # print(number) # 160 """
def frenet_optimal_traj(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd):
    
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    # fplist = check_paths(fplist, ob)
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path



def get_ego_traj():
    x_ego = []
    y_ego = []
    heading_ego = []
    vx_ego = []
    vy_ego = []
    DT = 0.1
    s0,c_speed,c_accel,c_d,dot_d,ddot_d,c_d_d,c_d_dd = cartesian2Frenet(x,y,theta,v,a,kappa,s_nearest,x_nearest,y_nearest,theta_nearest,k_nearest,k_r_prime)

    csp = cubic_spline_planner.CubicSpline2D(valid_extended[:, 0], valid_extended[:, 1])
    # print("开始循环")
    for loop in range(6):
        best_path = frenet_optimal_traj(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd)

        s0 = best_path.s[10]
        c_d = best_path.d[10]
        c_d_d = best_path.d_d[10]
        c_d_dd = best_path.d_dd[10]
        c_speed = best_path.s_d[10]
        c_accel = best_path.s_dd[10]

        x_ego.append(best_path.x[:10])
        y_ego.append(best_path.y[:10])
        heading_ego.append(best_path.yaw[:10])
        # print(time)
        # print(best_path.x)
        # if time <=5:
        for i in range(10):
            dx = best_path.x[i + 1] - best_path.x[i]
            dy = best_path.y[i + 1] - best_path.y[i]
            vx_ego.append(dx/DT)
            vy_ego.append(dy/DT)

    return x_ego,y_ego,heading_ego,vx_ego,vy_ego

from val_for_MCTS import qc_for_state
import copy
import torch
max_t = 60
import torch.nn.functional as F

x_ego,y_ego,heading_ego,vx_ego,vy_ego = get_ego_traj()
x_ego = [item for sublist in x_ego for item in sublist]
y_ego = [item for sublist in y_ego for item in sublist]
heading_ego = [item for sublist in heading_ego for item in sublist]

def smooth_heading(angles, window_size=5):
    """
    使用滑动平均对转向角进行平滑。
    
    参数:
    angles: 转向角数组 (1D)
    window_size: 滑动窗口大小，默认为5
    
    返回:
    smoothed_angles: 平滑后的转向角数组
    """
    smoothed_angles = np.convolve(angles, np.ones(window_size) / window_size, mode='same')
    return smoothed_angles

def get_forecasted_traj():
    forecasted_all = np.empty((max_t,nobjs,5))
    heading_all = np.empty((110,nobjs-1))
    # heading_all = np.empty((10,nobjs-1))
    for n in range(nobjs-1):
        heading_all[:50,n] = data['agent']['heading'][n][:50]
    h_s = copy.deepcopy(data)
    h_sp = copy.deepcopy(h_s)
    for j in range(0,6): # max_t
        qc_output = qc_for_state(h_s)
        eval_mask=3
        pi=qc_output['pi']
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        pi_eval = pi_eval.cpu().numpy()
        # max_value = np.max(pi_eval)
        # m_p_i = np.argwhere(pi_eval == max_value).flatten()
        random_index = np.random.choice([0, 1, 2, 3, 4, 5], p=pi_eval.ravel())
        for n in range(nobjs-1):
            theta=h_s['agent']['heading'][n][49]
            cos=math.cos(theta)
            sin=math.sin(theta)
            origin=h_s['agent']['position'][n][49]
            loc_refine_pos=qc_output['loc_refine_pos'][n].cpu()

            x=loc_refine_pos[random_index,:10,0]*cos-loc_refine_pos[random_index,:10,1]*sin+origin[0].unsqueeze(0) # loc_refine_pos[6,60,2]
            y=loc_refine_pos[random_index,:10,0]*sin+loc_refine_pos[random_index,:10,1]*cos+origin[1].unsqueeze(0) # [random_index,0,0]
            
            # heading = qc_output['loc_refine_head'][n,random_index,:10,0].cpu()+theta.unsqueeze(0) # [23,6,60,1]

            h_sp['agent']['position'][n,:40,:]=h_s['agent']['position'][n,10:50,:]
            h_sp['agent']['position'][n,40:50,0]=x
            h_sp['agent']['position'][n,40:50,1]=y
            h_sp['agent']['position'][n,50:100,:]=h_s['agent']['position'][n,60:,:]
            h_sp['agent']['position'][n,100:,:]=h_s['agent']['position'][n,109,:].unsqueeze(0)

            v_x = (h_sp['agent']['position'][n,40:50,0]-h_sp['agent']['position'][n,39:49,0])/0.1
            v_y = (h_sp['agent']['position'][n,40:50,1]-h_sp['agent']['position'][n,39:49,1])/0.1

            h_sp['agent']['velocity'][n,:40,:]=h_s['agent']['velocity'][n,10:50,:]
            h_sp['agent']['velocity'][n,40:50,0]=v_x
            h_sp['agent']['velocity'][n,40:50,1]=v_y
            h_sp['agent']['velocity'][n,50:100,:]=h_s['agent']['velocity'][n,60:,:]
            h_sp['agent']['velocity'][n,100:,:]=h_s['agent']['velocity'][n,109,:].unsqueeze(0)

            # for hn in range(10):
            #     heading_all[hn,n] = compute_heading(v_x[hn], v_y[hn])
            for hn in range(10):
                speed = math.sqrt(v_x[hn]**2 + v_y[hn]**2)
                min_speed_threshold = 0.2
                if speed > min_speed_threshold:
                    # print(v_x)
                    # print(v_y)
                    heading_all[50+j*10+hn,n]= compute_heading(v_x[hn], v_y[hn])
                else:
                    heading_all[50+j*10+hn,n] = heading_all[50+j*10+hn-1,n]

            h_sp['agent']['heading'][n,:40]= h_s['agent']['heading'][n,10:50]
            h_sp['agent']['heading'][n,40:50]= torch.FloatTensor(heading_all[50+j*10:50+j*10+10,n])
            # torch.FloatTensor(heading_all[:,n])
            # qc_output['loc_refine_head'][n,random_index,:10,0].cpu()+theta.unsqueeze(0)   
            # torch.FloatTensor(heading)
            h_sp['agent']['heading'][n,50:100]=h_s['agent']['heading'][n,60:]
            h_sp['agent']['heading'][n,100:]=h_s['agent']['heading'][n,109].unsqueeze(0)
        
            forecasted_all[j*10:j*10+10,n,0]=x # j,n,0
            forecasted_all[j*10:j*10+10,n,1]=y # j,n,0
            # forecasted_all[j*10:j*10+10,n,2]=heading_all[:,n]
            forecasted_all[j*10:j*10+10,n,3]=v_x
            forecasted_all[j*10:j*10+10,n,4]=v_y
        
        for i in range(10):

            h_sp['agent']['position'][-1,:49,:]=h_s['agent']['position'][-1,1:50,:]
            h_sp['agent']['position'][-1,49,0]=torch.tensor(x_ego[j*10+i])
            h_sp['agent']['position'][-1,49,1]=torch.tensor(y_ego[j*10+i])
            h_sp['agent']['position'][-1,50:109,:]=h_s['agent']['position'][-1,51:,:]
            h_sp['agent']['position'][-1,109,:]=h_s['agent']['position'][-1,109,:]

            h_sp['agent']['heading'][-1,:49]=h_s['agent']['heading'][-1,1:50]
            h_sp['agent']['heading'][-1,49]=torch.tensor(heading_ego[j*10+i])
            h_sp['agent']['heading'][-1,50:109]=h_s['agent']['heading'][-1,51:]
            h_sp['agent']['heading'][-1,109]=h_s['agent']['heading'][-1,109]
        
            h_sp['agent']['velocity'][-1,:49,:]=h_s['agent']['velocity'][-1,1:50,:]
            h_sp['agent']['velocity'][-1,49,0]=torch.tensor(vx_ego[j*10+i])
            h_sp['agent']['velocity'][-1,49,1]=torch.tensor(vy_ego[j*10+i])
            h_sp['agent']['velocity'][-1,50:109,:]=h_s['agent']['velocity'][-1,51:,:]
            h_sp['agent']['velocity'][-1,109,:]=h_s['agent']['velocity'][-1,109,:]

            forecasted_all[j*10+i,-1,0] = x_ego[j*10+i]
            forecasted_all[j*10+i,-1,1] = y_ego[j*10+i]
            forecasted_all[j*10+i,-1,2] = heading_ego[j*10+i]
            forecasted_all[j*10+i,-1,3] = vx_ego[j*10+i]
            forecasted_all[j*10+i,-1,4] = vy_ego[j*10+i]
            h_s = copy.deepcopy(h_sp)

    forecasted_all[:,:(nobjs-1),2] = heading_all[50:,:]
    headings = forecasted_all[:,6,2]  # 第三列为转向角
    smoothed_headings = smooth_heading(headings, window_size=5)
    forecasted_all[:,6,2] = smoothed_headings
    f_save = open('/home/niutian/原data/planner_result/forecasted_all_scene7_06.pkl', 'wb')  
    pickle.dump(forecasted_all, f_save)

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

if __name__ == '__main__':
    get_forecasted_traj()
# if __name__ == '__main__':
#     x_ego,y_ego,heading_ego,vx_ego,vy_ego = get_ego_traj()
#     print(x_ego)


# # MAX_SPEED = 33.33  # maximum speed [m/s]
# # MAX_ACCEL = 8.0  # maximum acceleration [m/ss]
# # MAX_CURVATURE = 1  # maximum curvature [1/m]
# # MAX_ROAD_WIDTH = 3.0 # maximum road width [m]
# # D_ROAD_W = 1.0  # road width sampling length [m]
# # DT = 0.1  # time tick [s]  影响曲线平滑程度的关键因素 
# # MAX_T = 10.0  # max prediction time [m]
# # MIN_T = 6.0  # min prediction time [m]
# # TARGET_SPEED = 8.0  # target speed [m/s]
# # D_T_S = 1  # target speed sampling length [m/s]
# # N_S_SAMPLE = 2  # sampling number of target speed
# # ROBOT_RADIUS = 2.0  # robot radius [m] for check collision


# # def coordinate_remapping(path_d,x_list,y_list,sample,x0_g_v):
# #     xy_stack = np.transpose(np.array([x_list,y_list])) - x0_g_v

# #     d = np.linalg.norm(xy_stack,ord=2, axis=1)
# #     # print("d=",d)
# #     min_index = np.argmin(d)

# #     s_map = sample[min_index]
# #     ey_map = d[min_index]

# #     theta_r = path_d.get_theta_r(s_map)
# #     sign = (x0_g_v[1]-y_list[min_index])*np.cos(theta_r) - (x0_g_v[0]-x_list[min_index])*np.sin(theta_r)
# #     if sign > 0:
# #         pass
# #     elif sign <0:
# #         ey_map = -ey_map
# #     return s_map,ey_map