import numpy as np

# 假设 forecasted_all 是已经填充好的，包含自车和他车的预测状态
# forecasted_all的结构为 [time_steps, nobjs, features]
# 其中 features 0:x, 1:y, 2:heading, 3:v_x, 4:v_y

# def calculate_comfort_cost(forecasted_self):
#     # 使用速度向量的大小变化作为舒适度指标
#     v_magnitude = np.sqrt(forecasted_self[:,3]**2 + forecasted_self[:,4]**2)
#     comfort_cost = np.sum(np.abs(np.diff(v_magnitude)))
#     return comfort_cost
def get_details_original(forecasted_all,nobjs):
    
    # 提取 x 轴位置、y 轴位置、转向角、x 轴速度、y 轴速度
    x = forecasted_all[:, :, 0]
    y = forecasted_all[:, :, 1]
    v_x = forecasted_all[:, :, 2]
    v_y = forecasted_all[:, :, 3]
    delta = forecasted_all[:, :, 4]

    # 计算加速度
    # 假设加速度是通过速度变化率计算得到的
    alpha_x = np.diff(v_x, axis=0, prepend=0)
    alpha_y = np.diff(v_y, axis=0, prepend=0)
    alpha = np.sqrt(alpha_x**2 + alpha_y**2)

    # y 轴速度
    dot_y = v_y

    # 计算车辆之间的 x 轴和 y 轴距离的平均值
    delta_x = np.zeros((10, nobjs))
    delta_y = np.zeros((10, nobjs))
    for t in range(10):
        for i in range(nobjs):
            delta_x[t, i] = np.mean([x[t, j] - x[t, i] for j in range(nobjs) if j != i])
            delta_y[t, i] = np.mean([y[t, j] - y[t, i] for j in range(nobjs) if j != i])

    return alpha, delta, dot_y, x, y, delta_x, delta_y

def get_details_continuous(history_state,nobjs):
    # history_state是data【nobjs,110,3】的形式
    # 提取 x 轴位置、y 轴位置、转向角、x 轴速度、y 轴速度
    x = np.transpose(history_state['agent']['position'][:, 40:50, 0], (1, 0)) # 交换第 1 和第 0 维度  维度从（19，10）变为（10，19）
    y = np.transpose(history_state['agent']['position'][:, 40:50, 1], (1, 0))
    v_x = np.transpose(history_state['agent']['velocity'][:, 40:50, 0], (1, 0))
    v_y = np.transpose(history_state['agent']['velocity'][:, 40:50, 1], (1, 0))
    delta = np.transpose(history_state['agent']['heading'][:, 40:50], (1, 0))

    # 计算加速度
    # 假设加速度是通过速度变化率计算得到的
    alpha_x = np.diff(v_x, axis=0, prepend=0)
    alpha_y = np.diff(v_y, axis=0, prepend=0)
    alpha = np.sqrt(alpha_x**2 + alpha_y**2)

    # y 轴速度
    dot_y = v_y

    # 计算车辆之间的 x 轴和 y 轴距离的平均值
    delta_x = np.zeros((10, nobjs))
    delta_y = np.zeros((10, nobjs))
    for t in range(10):
        for i in range(nobjs):
            delta_x[t, i] = np.mean([x[t, j] - x[t, i] for j in range(nobjs) if j != i])
            delta_y[t, i] = np.mean([y[t, j] - y[t, i] for j in range(nobjs) if j != i])

    return alpha, delta, dot_y, x, y, delta_x, delta_y


def cost_all(history_state,nobjs):

    alpha, delta, dot_y, x, y, delta_x, delta_y = get_details_continuous(history_state,nobjs)

    # 参数
    v0 = 31  #15
    W = 3.7
    w = 2.0
    bar_alpha = 4
    bar_alpha_neg = -5
    k4 = 15.0
    k6 = 3.0
    k7_x = 2.0
    k7_y = 20.0
    k8_x = 5.0
    k8_y = 20.0
    l7_x = 1.0
    l7_y = 1.0
    l8_x = 10.0
    l8_y = 2.0

    # 权重
    weights = np.array([1.0, 0.01, -1.5, 1.0, -0.3, -24.0, -2.0, -14.0])

    # 效用函数
    phi1 = 1 - ((np.transpose(history_state['agent']['heading'][:, 40:50], (1, 0)) - v0) / v0) ** 2
    phi1 = np.array(phi1) 
    phi2 = (alpha - np.roll(alpha, 1, axis=0)) ** 2
    phi3 = (delta - np.roll(delta, 1, axis=0)) ** 2
    phi3 = np.array(phi3) 
    phi4 = np.log(1 + np.exp(k4 * (alpha - bar_alpha))) + np.log(1 + np.exp(-k4 * (alpha - bar_alpha_neg)))
    # phi5 = np.minimum((dot_y - (W / 2) ** 2) / (3 * W ** 4 / 4), 1)
    # phi6 = 1 / (1 + np.exp(-k6 * (y - (W + w / 2))))
    # phi7 = 1 / (1 + np.exp(-k7_x * (x + l7_x))) * 1 / (1 + np.exp(-k7_y * (dot_y - l7_y)))
    # 限制 exponent 的输入范围在 [-500, 500] 之间，避免溢出
    exp_input_x = np.clip(-k8_x * (delta_x - l8_x), -500, 500)
    exp_input_y = np.clip(-k8_y * (delta_y - l8_y), -500, 500)
    phi8 = 1 / (1 + np.exp(exp_input_x)) + 1 / (1 + np.exp(exp_input_y))
    # phi8 = 1 / (1 + np.exp(-k8_x * (delta_x - l8_x))) + 1 / (1 + np.exp(-k8_y * (delta_y - l8_y)))
        # 检查是否有无效值 NaN 或 Inf
    if (np.isnan(phi1).any() or np.isinf(phi1).any() or
        np.isnan(phi2).any() or np.isinf(phi2).any() or
        np.isnan(phi3).any() or np.isinf(phi3).any() or
        np.isnan(phi4).any() or np.isinf(phi4).any() or
        np.isnan(phi8).any() or np.isinf(phi8).any() or
        np.isnan(weights).any() or np.isinf(weights).any()):
        
        print("Invalid value encountered in the input data.")
        return np.nan  # 直接返回 NaN 或者标记无效值

    # 总成本函数
    cost = np.sum(weights[0] * phi1 + weights[1] * phi2 + weights[2] * phi3 + weights[3] * phi4 + weights[7] * phi8)
    # + weights[4] * phi5 + weights[5] * phi6 + weights[6] * phi7

    return cost

# 缺少偏移车道线那项
def cost_all_original(forecasted_all,nobjs):

    alpha, delta, dot_y, x, y, delta_x, delta_y = get_details_continuous(forecasted_all,nobjs)

    # 参数
    v0 = 31  #15
    W = 3.7
    w = 2.0
    bar_alpha = 4
    bar_alpha_neg = -5
    k4 = 15.0
    k6 = 3.0
    k7_x = 2.0
    k7_y = 20.0
    k8_x = 5.0
    k8_y = 20.0
    l7_x = 1.0
    l7_y = 1.0
    l8_x = 10.0
    l8_y = 2.0

    # 权重
    weights = [1.0, 0.01, -1.5, 1.0, -0.3, -24.0, -2.0, -14.0]

    # 效用函数
    phi1 = 1 - ((forecasted_all[:, :, 4] - v0) / v0) ** 2
    phi2 = (alpha - np.roll(alpha, 1, axis=0)) ** 2
    phi3 = (delta - np.roll(delta, 1, axis=0)) ** 2
    phi4 = np.log(1 + np.exp(k4 * (alpha - bar_alpha))) + np.log(1 + np.exp(-k4 * (alpha - bar_alpha_neg)))
    # phi5 = np.minimum((dot_y - (W / 2) ** 2) / (3 * W ** 4 / 4), 1)
    # phi6 = 1 / (1 + np.exp(-k6 * (y - (W + w / 2))))
    # phi7 = 1 / (1 + np.exp(-k7_x * (x + l7_x))) * 1 / (1 + np.exp(-k7_y * (dot_y - l7_y)))
    # 限制 exponent 的输入范围在 [-500, 500] 之间，避免溢出
    exp_input_x = np.clip(-k8_x * (delta_x - l8_x), -500, 500)
    exp_input_y = np.clip(-k8_y * (delta_y - l8_y), -500, 500)
    phi8 = 1 / (1 + np.exp(exp_input_x)) + 1 / (1 + np.exp(exp_input_y))
    # phi8 = 1 / (1 + np.exp(-k8_x * (delta_x - l8_x))) + 1 / (1 + np.exp(-k8_y * (delta_y - l8_y)))
    # print("phi1:",phi1.shape)
    # print("phi2:",phi2.shape)
    # print("phi3:",phi3.shape)
    # print("phi4:",phi4.shape)
    # print("phi8:",phi8.shape)
        # 检查是否有无效值 NaN 或 Inf
    if (np.isnan(phi1).any() or np.isinf(phi1).any() or
        np.isnan(phi2).any() or np.isinf(phi2).any() or
        np.isnan(phi3).any() or np.isinf(phi3).any() or
        np.isnan(phi4).any() or np.isinf(phi4).any() or
        np.isnan(phi8).any() or np.isinf(phi8).any() or
        np.isnan(weights).any() or np.isinf(weights).any()):
        
        print("Invalid value encountered in the input data.")
        return np.nan  # 直接返回 NaN 或者标记无效值
    # 总成本函数
    cost = np.sum(weights[0] * phi1 + weights[1] * phi2 + weights[2] * phi3 + weights[3] * phi4 + weights[7] * phi8)
    # + weights[4] * phi5 + weights[5] * phi6 + weights[6] * phi7

    return cost

















def calculate_velocity_magnitude(v_x, v_y):
    return np.sqrt(v_x**2 + v_y**2)

def calculate_acceleration(v_x, v_y):
    # 计算速度向量的大小变化作为加速度的近似
    acc_x = np.diff(v_x, prepend=v_x[0])
    acc_y = np.diff(v_y, prepend=v_y[0])
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2)
    return acc_magnitude


def comfort_checks_cost(forecasted_all):  # 下面用的时候其实暂时只算了forcasted_others
    MAX_SPEED = 33.33 # 假设最大速度限制，单位：米/秒 (120公里/小时)
    MAX_ACCEL = 3 # 最大加速度限制，单位：米/秒^2
    # MAX_CURVATURE = 0.1 # 最大曲率限制，示例值
    comfort_cost = 0
    forecasted_all_np = np.array(forecasted_all)
    for nobj in range(forecasted_all_np.shape[1]):
        x = forecasted_all_np[:,nobj,0]
        y = forecasted_all_np[:,nobj,1]
        v_x = forecasted_all_np[:,nobj,3]
        v_y = forecasted_all_np[:,nobj,4]

        speed = calculate_velocity_magnitude(v_x, v_y)
        accel = calculate_acceleration(v_x, v_y)
        # curvature = calculate_curvature(x, y)

        max_speed_check = np.all(speed <= MAX_SPEED)
        max_accel_check = np.all(accel <= MAX_ACCEL)
        # max_curvature_check = np.all(curvature <= MAX_CURVATURE)
        if max_speed_check == False or max_accel_check == False:
            comfort_cost += 0.05
    return comfort_cost

def calculate_collision_cost(forecasted_self, forecasted_others):
    # 计算最近的他车距离
    min_distance = np.inf
    for obj in forecasted_others:
        distances = np.sqrt((forecasted_self[:,0] - obj[:,0])**2 + (forecasted_self[:,1] - obj[:,1])**2)
        min_distance = min(min_distance, np.min(distances))
    
    # 简化的碰撞成本计算，距离越小，成本越高
    if min_distance < 1e-6: # 防止除以零
        collision_cost = np.inf
    else:
        collision_cost = 1 / min_distance
    return collision_cost



# def cost_all(forecasted_all,nobjs):

#     # 分离自车和他车的数据
#     forecasted_self = forecasted_all[:, -1, :]
#     forecasted_others = [forecasted_all[:, i, :] for i in range(nobjs-1)]

#     # 计算成本
#     comfort_cost = comfort_checks_cost(forecasted_others)
#     collision_cost = calculate_collision_cost(forecasted_self, forecasted_others)
#     # cost_all = comfort_cost + collision_cost
#     return comfort_cost,collision_cost

# print(f"Comfort Cost: {comfort_cost}")
# print(f"Collision Risk Cost: {collision_cost}")
