import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
FD003_train_x = pickle.load(f_read)
f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD003_train_y.pkl', 'rb')
FD003_train_y = pickle.load(f_read)

f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD003_test_last_x.pkl', 'rb')
FD003_test_x = pickle.load(f_read)
f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD003_test_last_y.pkl', 'rb')
FD003_test_y = pickle.load(f_read)

# 将特征和标签数据组合成一个 DataFrame
# train_data = pd.DataFrame(FD003_train_x)
# train_data['label'] = FD003_train_y

# test_data = pd.DataFrame(FD003_test_x)
# test_data['label'] = FD003_test_y

# 为每个时间步长计算相关性矩阵并绘制热力图
# fig, axes = plt.subplots(6, 5, figsize=(20, 15))
# axes = axes.flatten()

# for i in range(30):
#     data_2d = pd.DataFrame(FD003_train_x[:, i, :], columns=[f'Feature {j+1}' for j in range(14)])
#     corr_matrix = data_2d.corr()
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[i])
#     axes[i].set_title(f'Time Step {i+1}')
#     axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
#     axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)

# # 移除多余的子图
# for j in range(30, len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_correlation.png')

# 为每个时间步长绘制图
# 创建3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 将数据绘制到3D图中
for t in range(30):
    for f in range(14):
        ax.scatter(t, f, FD003_train_x[:, t, f], alpha=0.6, c=FD003_train_y.flatten(), cmap='viridis')

# 设置轴标签
ax.set_xlabel('Time Step')
ax.set_ylabel('Feature Index')
ax.set_zlabel('Feature Value')

# 设置轴刻度范围
ax.set_xticks(np.arange(30))
ax.set_yticks(np.arange(14))

plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD003_correlation.png')

