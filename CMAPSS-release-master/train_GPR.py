import numpy as np
import matplotlib.pyplot as plt 
import time
import torch
from dataset import *
from sklearn.metrics import mean_squared_error
# from train_LSTM_RFR import *



train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

test_egine1 = get_test_engine(dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/', sub_dataset='FD003', max_rul=125, seq_length=30, smooth_rate=128, engine_id=1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C




    # 定义高斯过程的核函数
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e3))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100, length_scale_bounds=(1e-1, 1e4))
    # 设置Matérn核，nu控制平滑性，length_scale控制相关性的距离范围
    # kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10), nu=1.5) # matern_
    # 设置Rational Quadratic核，alpha参数控制不同长度尺度的混合程度
kernel = C(1.0, (1e-7, 1e3)) * RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-1, 10), alpha=1.0) # rq_  #-8太大了，尝试-6(15.82)，尝试-5(16.0)

# 创建高斯过程回归模型
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1.0)  # 增加n_restarts_optimizer和alpha


features = []
labels_list = []

for inputs, labels in train_loader:
    # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
    features.append(inputs.numpy())
    labels_list.append(labels.numpy())

train_input = np.concatenate(features, axis=0)
train_labels = np.concatenate(labels_list, axis=0)
# 使用LSTM特征和训练标签训练GPR模型
start_time = time.time()
gpr.fit(train_input[:100], train_labels[:100])
end_time = time.time()
print(f"GPR Prediction completed in {end_time - start_time:.2f} seconds.")



def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return torch.sum(torch.where(diff < 0, torch.exp(-diff/13)-1, torch.exp(diff/10)-1))


x_test, y_test = next(iter(test_loader)) # torch.Size([500, 30, 14]) y_test torch.Size([500, 1])
y_test = (y_test.reshape(-1)) * 125
# print(test_lstm_features.size)
rul_pred, gpr_std = gpr.predict(x_test, return_std=True) # (500, 1) [[0.55744501][0.54677675][0.53615835][0.52466257][0.51270392][0.7166863 ][0.70877164]
rul_pred = rul_pred.reshape(-1)  # (497,1) -- (497,)
rul_pred *= 125
preds_for_each_engine = np.array_split(rul_pred, len(num_test_windows)) # list
# print("reds_for_each_engine1",preds_for_each_engine)
gpr_std = gpr_std.reshape(-1)  # (497,1) -- (497,)
gpr_std *= 125
std_for_each_engine = np.array_split(gpr_std, len(num_test_windows))

y_test_windows = np.array_split(y_test, len(num_test_windows))
y_test_means = [item.mean() for item in y_test_windows]
index = np.argsort(-np.array(y_test_means))
y_test = np.array(y_test_means)[index]
mean_pred_for_each_engine = [item.sum() / len(item) for item in preds_for_each_engine]
mean_pred_for_each_engine = np.take(mean_pred_for_each_engine, index, axis=0)
mean_pred_for_each_engine = np.floor(mean_pred_for_each_engine) # 向下取整
# print("mean_pred_for_each_engine",mean_pred_for_each_engine)
mean_std_for_each_engine = [item.sum() / len(item) for item in std_for_each_engine]
mean_std_for_each_engine = np.take(mean_std_for_each_engine, index, axis=0)
mean_std_for_each_engine = np.floor(mean_std_for_each_engine) # 向下取整
# 使用测试特征进行预测

rf_rmse = mean_squared_error(y_test, mean_pred_for_each_engine, squared=False).item()
score = compute_s_score(y_test, mean_pred_for_each_engine)
print(f'Test RMSE: {rf_rmse}')
print(f'Test score: {score}')

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual RUL')
plt.plot(mean_pred_for_each_engine, label='Predicted RUL')
plt.fill_between(range(len(y_test)), mean_pred_for_each_engine - 1.96*mean_std_for_each_engine, mean_pred_for_each_engine + 1.96*mean_std_for_each_engine, alpha=0.4)
plt.xlabel('Sample')
plt.ylabel('RUL')
plt.legend()
plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_GPR.png')
plt.show()

    # 调用此函数展示分布
    # 假设`engine`和`differences`是前面代码中收集的引擎索引和它们的差值
    # plot_large_diff_distribution(engine, [test_lstm_features[i] for i in engine])

# def GPR_test(test_loader):
#     gpr = GPR_train(train_loader)
#     test_lstm_features,test_labels = extract_lstm_features(model_lstm, test_loader)
#     # print(test_lstm_features.shape)  # (204,100)
#     # print(test_lstm_features)
#     # 使用测试特征进行预测
#     gpr_predictions, gpr_std = gpr.predict(test_lstm_features, return_std=True)
#     # print(gpr_predictions.shape)
#     # print(gpr_predictions)
#     gpr_predictions = gpr_predictions.flatten()  # 确保是一维
#     # print(gpr_std)
#     gpr_std = gpr_std.flatten()  # 确保是一维
#     test_labels = test_labels.flatten()  # 确保是一维
#     # 计算测试RMSE
#     gpr_rmse = mean_squared_error(test_labels, gpr_predictions*125, squared=False)
#     print(f'GPR Test RMSE: {gpr_rmse}')

#     # # 可视化预测结果和实际值
#     plt.figure(figsize=(12, 6))
#     plt.plot(test_labels, label='Actual RUL')
#     plt.plot(gpr_predictions*125, label='Predicted RUL')
#     plt.fill_between(range(len(test_labels)), gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125, alpha=0.4)
#     plt.xlabel('Sample')
#     plt.ylabel('RUL')
#     plt.legend()
#     plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_engine_1_GPR.png')
#     # plt.show()

#     return gpr_predictions,gpr_std






