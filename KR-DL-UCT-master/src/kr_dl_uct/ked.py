"""

这段代码实现了一个简单的核密度估计器（KDE，Kernel Density Estimation）。以下是各部分的详细解析：
1. 核心数据结构
m_obs: 存储二维空间中的观测点 (x, y)。这些观测点是 KDE 的数据来源，用于计算密度。
2. 方法功能
add_ob:
用于向 m_obs 中添加新的观测点。
在实际使用中，这些点是用于密度估计的数据。
kernel:
静态方法，计算两点之间的核密度。
使用高斯核函数： K(dx,dy)=exp⁡(−12(dx2varx+dy2vary))K(dx, dy) = \exp\left(-\frac{1}{2} \left(\frac{dx^2}{\text{var}_x} + \frac{dy^2}{\text{var}_y}\right)\right)K(dx,dy)=exp(−21(varxdx2+varydy2))
参数 var_x 和 var_y 控制核的宽度，可以根据应用需求调整。
eval:
对目标点 (x, y) 进行密度估计。
将目标点与所有观测点计算核密度，并将结果相加，得到总密度。

"""

import math
from typing import List, Tuple

# 假设 var_x 和 var_y 是全局定义的变量（从 "setting.h" 引入）
var_x = 0.145 * 0.5  # 示例值，可根据需求调整
var_y = 0.145 * 2.0  # 示例值，可根据需求调整

class KDE:
    def __init__(self):
        """初始化 KDE 对象，准备存储观测值 (x, y)"""
        self.m_obs: List[Tuple[float, float]] = []

    def add_ob(self, x: float, y: float):
        """添加一个观测点 (x, y)"""
        self.m_obs.append((x, y))

    @staticmethod
    def kernel(dx: float, dy: float) -> float:
        """
        静态方法，计算给定偏移量 (dx, dy) 的核密度估计值。
        使用高斯核函数，公式为：
        exp(-0.5 * ((dx^2 / var_x) + (dy^2 / var_y)))
        """
        return math.exp(-0.5 * ((math.pow(dx, 2.0) / var_x) + (math.pow(dy, 2.0) / var_y)))

    def eval(self, x: float, y: float) -> float:
        """
        对给定点 (x, y) 计算核密度估计值。
        公式为核函数值的累加：
        KDE(x, y) = Σ kernel(x - x_i, y - y_i) for all (x_i, y_i) in m_obs
        """
        return sum(self.kernel(x - x_i, y - y_i) for x_i, y_i in self.m_obs)

# 使用示例
kde = KDE()
kde.add_ob(1.0, 2.0)  # 添加观测点
kde.add_ob(3.0, 4.0)

result = kde.eval(2.0, 3.0)  # 计算给定点 (2.0, 3.0) 的核密度估计值
print("核密度估计值:", result)