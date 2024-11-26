# 在椭圆内随机生成点


import math
import random

class EllipseSampleUtils:
    @staticmethod
    def rand_zero_to_one():
        """生成一个 [0, 1) 范围内的随机数"""
        return random.uniform(0.0, 1.0)

    @staticmethod
    def generate_theta(r_x, r_y):
        """
        根据给定的椭圆半径生成随机角度 theta，确保在椭圆上均匀分布。
        """
        u = EllipseSampleUtils.rand_zero_to_one() / 4.0
        theta = math.atan(r_y / r_x * math.tan(2 * math.pi * u))

        v = EllipseSampleUtils.rand_zero_to_one()
        if v < 0.25:
            return theta
        elif v < 0.5:
            return math.pi - theta
        elif v < 0.75:
            return math.pi + theta
        else:
            return -theta

    @staticmethod
    def radius(r_x, r_y, theta):
        """
        根据给定的角度 theta 和椭圆的半径计算极坐标下的半径。
        """
        return r_x * r_y / math.sqrt((r_y * math.cos(theta)) ** 2 + (r_x * math.sin(theta)) ** 2)

    @staticmethod
    def uniform_random_point(r_x, r_y):
        """
        在椭圆内均匀生成一个随机点。
        """
        random_theta = EllipseSampleUtils.generate_theta(r_x, r_y)
        max_radius = EllipseSampleUtils.radius(r_x, r_y, random_theta)
        random_radius = max_radius * math.sqrt(EllipseSampleUtils.rand_zero_to_one())

        random_point = [
            random_radius * math.cos(random_theta),
            random_radius * math.sin(random_theta)
        ]
        return random_point


# 使用示例：
r_x = 5  # 椭圆 x 方向的半径
r_y = 3  # 椭圆 y 方向的半径
random_point = EllipseSampleUtils.uniform_random_point(r_x, r_y)
print("随机点:", random_point)