from math import *
import pandas as pd
import pickle
from frenet_optimal_trajectory import *
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from pathlib import Path
from av2.utils.io import read_json_file
from av2.map.map_api import ArgoverseStaticMap
from get_data_for_val import get_data



# 计算给定点到曲线的最短距离
f_read = open('/home/niutian/原data/code/QCNet-main/centerline_7.pkl', 'rb')
extended_filtered_points = pickle.load(f_read)
# print("centerline的坐标",extended_filtered_points)
s = np.zeros(extended_filtered_points.shape[0])
s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(extended_filtered_points, axis=0)**2, axis=1)))
cs_x = CubicSpline(s, extended_filtered_points[:, 0])
cs_y = CubicSpline(s, extended_filtered_points[:, 1])
csp = cubic_spline_planner.CubicSpline2D(extended_filtered_points[:, 0], extended_filtered_points[:, 1])

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
    d=np.sign((y-y_r)*math.cos(theta_r)-(x-x_r)*math.sin(theta_r))*math.sqrt((x-x_r)**2+(y-y_r)**2)
    dot_d = v_x*math.sin(delta_theta)
    ddot_d = a_x*math.sin(delta_theta)
    dot_s=v_x*math.cos(delta_theta)/one_kr_d
    d_=one_kr_d*math.tan(delta_theta)
    d__=-(k_r_*d+k_r*d_)*math.tan(delta_theta)+one_kr_d/(math.cos(delta_theta))**2*(k_x*one_kr_d/math.cos(delta_theta)-k_r)
    ddot_s = (a_x*math.cos(delta_theta)-dot_s**2*(d_*(k_x*one_kr_d/math.cos(delta_theta)-k_r)-(k_r_*d+k_r*d_)))/one_kr_d

    return s,dot_s,ddot_s,d,dot_d,ddot_d,d_,d__

def self_curvature(s):
    dx_ds = cs_x.derivative()(s)
    dy_ds = cs_y.derivative()(s)
    d2x_ds2 = cs_x.derivative().derivative()(s)
    d2y_ds2 = cs_y.derivative().derivative()(s)
    return (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)

def distance_to_curve(s, x0, y0):
    x = cs_x(s)
    y = cs_y(s)
    return np.sqrt((x - x0)**2 + (y - y0)**2)

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

def global_to_frenet(input, history):                                                                                 
    # x_start = np.array(input['agent']['position'][-1,49,0])
    # y_start = np.array(input['agent']['position'][-1,49,1])

    # v_x_start = input['agent']['velocity'][-1,49,0]
    # v_y_start = input['agent']['velocity'][-1,49,1]

    # heading_start = input['agent']['heading'][-1,49]

    # 使用数值优化找到最近点的s值
    result = minimize_scalar(distance_to_curve, args=(input[0], input[1]), bounds=(s[0], s[-1]), method='bounded')

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

    k_r = self_curvature(s_nearest)
    k_r_plus = self_curvature(s_plus)
    k_r_minus = self_curvature(s_minus)

    # 使用中心差分公式计算 k'_r
    k_r_prime = (k_r_plus - k_r_minus) / (2 * delta_s)

    x_p1 = history['agent']['position'][-1,48,0]
    y_p1 = history['agent']['position'][-1,48,1]

    v_x_p1 = history['agent']['velocity'][-1,48,0]
    v_y_p1 = history['agent']['velocity'][-1,48,1]

    heading_p1 = history['agent']['heading'][-1,48]

    x_p2 = history['agent']['position'][-1,47,0]
    y_p2 = history['agent']['position'][-1,47,1]

    v_x_p2 = history['agent']['velocity'][-1,47,0]
    v_y_p2 = history['agent']['velocity'][-1,47,1]

    heading_p2 = history['agent']['heading'][-1,47]

    # 示例点
    p1 = [x_p2, y_p2]
    p2 = [x_p1, y_p1]
    p3 = [input[0], input[1]]
    # 还需要获得曲率curvature k 和 加速度
    # 计算曲率
    curvature = calculate_curvature(p1, p2, p3)
    dt=0.1
    v2 = np.array([input[2], input[3]]) 
    v1 = np.array([v_x_p1, v_y_p1]) 
    acceleration = (v2 - v1) / dt

    x = input[0]
    y = input[1]
    v = math.sqrt(input[2]**2 + input[3]**2)  # 1.78859228093272
    a = math.sqrt(acceleration[0]**2 + acceleration[1]**2) # 0.5462798942874714
    theta = input[4]
    kappa = curvature
    s0,c_speed,c_accel,c_d,dot_d,ddot_d,c_d_d,c_d_dd = cartesian2Frenet(x,y,theta,v,a,kappa,s_nearest,x_nearest,y_nearest,theta_nearest,k_nearest,k_r_prime)

    return c_speed, c_accel, c_d, c_d_d, c_d_dd, s0

def initialize_frenet(input, history):                                                                                 

    # x_start = np.array(input['agent']['position'][-1,49,0])
    # y_start = np.array(input['agent']['position'][-1,49,1])

    # v_x_start = input['agent']['velocity'][-1,49,0]
    # v_y_start = input['agent']['velocity'][-1,49,1]

    # heading_start = input['agent']['heading'][-1,49]

    # 使用数值优化找到最近点的s值
    result = minimize_scalar(distance_to_curve, args=(input[0], input[1]), bounds=(s[0], s[-1]), method='bounded')

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

    k_r = self_curvature(s_nearest)
    k_r_plus = self_curvature(s_plus)
    k_r_minus = self_curvature(s_minus)

    # 使用中心差分公式计算 k'_r
    k_r_prime = (k_r_plus - k_r_minus) / (2 * delta_s)

    x_p1 = history['agent']['position'][-1,48,0]
    y_p1 = history['agent']['position'][-1,48,1]

    v_x_p1 = history['agent']['velocity'][-1,48,0]
    v_y_p1 = history['agent']['velocity'][-1,48,1]

    heading_p1 = history['agent']['heading'][-1,48]

    x_p2 = history['agent']['position'][-1,47,0]
    y_p2 = history['agent']['position'][-1,47,1]

    v_x_p2 = history['agent']['velocity'][-1,47,0]
    v_y_p2 = history['agent']['velocity'][-1,47,1]

    heading_p2 = history['agent']['heading'][-1,47]

    # 示例点
    p1 = [x_p2, y_p2]
    p2 = [x_p1, y_p1]
    p3 = [input[0], input[1]]
    # 还需要获得曲率curvature k 和 加速度
    # 计算曲率
    curvature = calculate_curvature(p1, p2, p3)
    dt=0.1
    v2 = np.array([input[2], input[3]]) 
    v1 = np.array([v_x_p1, v_y_p1]) 
    acceleration = (v2 - v1) / dt

    x = input[0]
    y = input[1]
    v = math.sqrt(input[2]**2 + input[3]**2)  # 1.78859228093272
    a = math.sqrt(acceleration[0]**2 + acceleration[1]**2) # 0.5462798942874714
    theta = input[4]
    kappa = curvature
    s0,c_speed,c_accel,c_d,dot_d,ddot_d,c_d_d,c_d_dd = cartesian2Frenet(x,y,theta,v,a,kappa,s_nearest,x_nearest,y_nearest,theta_nearest,k_nearest,k_r_prime)

    return c_speed, c_accel, c_d, c_d_d, c_d_dd, s0


def frenet_optimal_traj(target_speed, csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd):
    
    fplist = calc_frenet_paths(target_speed,c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    # fplist = check_paths(fplist, ob)
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path

# 每次只选一个best_path，然后通过修改不同的速度范围获得集合
# 需要返回s和sp的信息
# reward计算是用0.01那个时刻的还是1s时间段的？？
def get_ego_traj(s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, target_speed):

    # 第一次input和history没问题，是之后自车状态更新得不正确
    # 更新sp的初始化
    x_ego = []
    y_ego = []
    heading_ego = []
    vx_ego = []
    vy_ego = []

    # 把状态变为frenet可输入的形式
    # 原来的是用best_path直接更新
    # c_speed, c_accel, c_d, c_d_d, c_d_dd, s0 = global_to_frenet(input,history)


    best_path = frenet_optimal_traj(target_speed,csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd)

    # fplist = calc_frenet_paths(target_speed, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    # fplist = calc_global_paths(fplist, csp)
    # min_cost = float("inf")
    # best_path = None
    # 不是cost问题，因为fp到后面都很大，不管target_speed大小
    # 研究frenet path是否有迭代次数的限制
    # 如果不行则采用每1s一个决策
    # 研究通常的决策频率是多少
    # 尝试单独ego_optimal是否有类似问题，单独进行循环没有类似问题，可以循环60次，但此时参数设置是按frenet本身参数
    # 查看是否自车速度就被规划地很大，速度就有问题
    # for fp in fplist:
    #     if min_cost >= fp.cf:
    #         min_cost = fp.cf
    #         best_path = fp
    
    # 注意best path的选取方法
    s0 = best_path.s[:10]
    c_d = best_path.d[:10]
    c_d_d = best_path.d_d[:10]
    c_d_dd = best_path.d_dd[:10]
    c_speed = best_path.s_d[:10]
    c_accel = best_path.s_dd[:10]

    x_ego.append(best_path.x[1:11])
    y_ego.append(best_path.y[1:11])
    heading_ego.append(best_path.yaw[1:11])
    # print("best_path:",best_path.x[:])
    for i in range(1,11):
        dx = best_path.x[i + 1] - best_path.x[i]
        dy = best_path.y[i + 1] - best_path.y[i]
        vx_ego.append(dx/DT)
        vy_ego.append(dy/DT)
   
    return x_ego, y_ego, heading_ego, vx_ego, vy_ego, s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, 


def get_ego_traj_new(s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, target_speed):


    x_ego = []
    y_ego = []
    heading_ego = []
    vx_ego = []
    vy_ego = []

    for loop in range(6):
        best_path = frenet_optimal_traj(target_speed,csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd)
        if loop == 0:
            o_s0 = best_path.s[1]
            o_c_d = best_path.d[1]
            o_c_d_d = best_path.d_d[1]
            o_c_d_dd = best_path.d_dd[1]
            o_c_speed = best_path.s_d[1]
            o_c_accel = best_path.s_dd[1]
        s0 = best_path.s[10]
        c_d = best_path.d[10]
        c_d_d = best_path.d_d[10]
        c_d_dd = best_path.d_dd[10]
        c_speed = best_path.s_d[10]
        c_accel = best_path.s_dd[10]

        x_ego.append(best_path.x[1:11])
        y_ego.append(best_path.y[1:11])
        heading_ego.append(best_path.yaw[1:11])
        # print(time)
        # print(best_path.x)
        # if time <=5:
        for i in range(1,11):
            dx = best_path.x[i + 1] - best_path.x[i]
            dy = best_path.y[i + 1] - best_path.y[i]
            vx_ego.append(dx/DT)
            vy_ego.append(dy/DT)
   
    return x_ego, y_ego, heading_ego, vx_ego, vy_ego, o_s0, o_c_d, o_c_d_d, o_c_d_dd, o_c_speed, o_c_accel, 