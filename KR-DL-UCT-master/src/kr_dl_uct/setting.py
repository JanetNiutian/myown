# 定义了一些用于描述统计特性的常量（标准差和方差），以及动作维度和得分维度的常量。
# 这些常量可能被用在概率分布的计算中（例如，高斯分布），其中标准差和方差定义了分布的形状，可能用于表示动作或得分的随机分布范围。

import math

# 定义常量
action_dim = 3  # 动作的维度
score_dim = 17  # 得分的维度

# 标准差
std_x = 0.145 * 0.5
std_y = 0.145 * 2.0

# 方差
var_x = math.pow((0.145 * 0.5), 2.0)
var_y = math.pow((0.145 * 2.0), 2.0)

# 打印结果以验证
print("标准差 (std_x):", std_x)
print("标准差 (std_y):", std_y)
print("方差 (var_x):", var_x)
print("方差 (var_y):", var_y)