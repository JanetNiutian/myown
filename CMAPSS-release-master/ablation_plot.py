
import matplotlib.pyplot as plt
import numpy as np

# 假设数据 (可以替换为你的实际数据)
methods = ['LSTM', 'LSTMGPR']
test_sets = ['FD001', 'FD002', 'FD003', 'FD004']
indicators = ['RMSE', 'Score']

# 创建一个 (4, 2, 3) 形状的数组
# 这意味着有4个 '2x3' 的矩阵
data = np.array([
    [  # 第一个 2x3 矩阵
        [15.86190414428711,  14],
        [603.9562377929688,  500]
    ],
    [  # 第二个 2x3 矩阵
        [13.26401424407959,  11],
        [853.0657958984375,  500]
    ],
    [  # 第三个 2x3 矩阵
        [17.499143600463867,  14],
        [1457.9892578125,  500]
    ],
    [  # 第四个 2x3 矩阵
        [21.301881790161133,  15],
        [3775.96240234375,  500]
    ]
])

fig, ax1 = plt.subplots(figsize=(14, 6))
width = 0.35  # 柱宽

# x 坐标计算，每个测试集有两组数据（每种方法一组）
# x_base = np.arange(len(test_sets) * len(methods) * 2)
x_base = np.arange(len(test_sets) * len(methods) * 2) * (width * 1.5)

# 为了显示 Score，创建一个第二个y轴
ax2 = ax1.twinx()

# 绘制每个测试集的 RMSE 和 Score
for i in range(len(test_sets)):
    for j in range(len(methods)):
        # 计算当前方法的 x 坐标
        x_offset = i * len(methods) * 2 + j * 2
        rmses = data[i, 0, j]
        scores = data[i, 1, j]
        
        ax1.bar(x_base[x_offset], rmses, width, label=f'{test_sets[i]} {methods[j]} RMSE', color='blue')
        ax2.bar(x_base[x_offset] + width, scores, width, label=f'{test_sets[i]} {methods[j]} Score', color='orange')

# 设置图表标题和轴标签
ax1.set_xlabel('Test Set and Method Combination')
ax1.set_ylabel('RMSE', color='blue')
ax2.set_ylabel('Score', color='orange')
ax1.set_title('Comparison of RMSE and Score Across Different Test Sets')
# ax1.set_xticks(x_base + width / 2)
# ax1.set_xticklabels([f'{ts} {m}' for ts in test_sets for m in methods for _ in (0, 1)])  # 为每个条目生成两次标签

# 图例设置
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# 去除图例和横坐标标签
ax1.set_xticks([])
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)

plt.tight_layout()
plt.savefig('/data/niutian/code/CMAPSS-release-master/ablation.png')
plt.show()
