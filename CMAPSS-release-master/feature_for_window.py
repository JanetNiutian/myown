import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.ticker import MultipleLocator

f_read = open('/data/niutian/code/CMAPSS-release-master/FD004_train_x.pkl', 'rb')
FD004_train_x = pickle.load(f_read)
extracted_data = FD004_train_x[:1000, 0, :]

# 使用 Seaborn 颜色调色板
colors = sns.color_palette('husl', 14)
x_major_locator = MultipleLocator(100)
# 为每个特征绘制单独的图
for i in range(14):
    plt.figure(figsize=(14, 2))  # 设置较长的 x 轴
    plt.plot(extracted_data[:, i], label='train', color=colors[i], alpha=0.7)
    plt.title(f'Feature {i+1}')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    # 设置 x 轴范围，确保起始点和终止点都在边界上
    plt.ylim(-4,4)
    plt.xlim(0, len(extracted_data) - 1)
    plt.gca().xaxis.set_major_locator(x_major_locator)
    # 设置刻度线在框架内
    plt.tick_params(axis='both', which='both', direction='in')
    # 移除图例
    plt.legend().remove()
    plt.savefig(f'/data/niutian/code/CMAPSS-release-master/feature{i+1}_for_window.png')
    plt.show()

