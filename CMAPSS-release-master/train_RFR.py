import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    diff_tensor = torch.from_numpy(diff)
    score = torch.sum(torch.where(diff_tensor < 0, torch.exp(-diff_tensor/13)-1, torch.exp(diff_tensor/10)-1))
    return score.item()  # 使用.item()将单个元素的Tensor转换为Python的数字
    # diff = rul_pred - rul_true
    # return torch.sum(torch.where(diff < 0, torch.exp(-diff/13)-1, torch.exp(diff/10)-1))

def evaluate(y_test, true_pred):
    RMSE = mean_squared_error(y_test, true_pred, squared=False).item()
    score = compute_s_score(y_test, true_pred) # mean_pred_for_each_engine
    return RMSE, score

f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
FD003_train_x = pickle.load(f_read)# (19072, 30, 14) --sequence-len,--feature-num
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_y.pkl', 'rb')
FD003_train_y = pickle.load(f_read) # 19072
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_x.pkl', 'rb')
FD003_test_x = pickle.load(f_read)
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_y.pkl', 'rb')
FD003_test_y = pickle.load(f_read)

# 随机森林回归
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# 由于随机森林不能直接处理三维数据，需要先将其reshape
X_train_flat = FD003_train_x.reshape(-1, 30*14)
X_test_flat = FD003_test_x.reshape(-1, 30*14)
start_time = time.time()
rf_model.fit(X_train_flat, FD003_train_y.ravel())
end_time = time.time()
# 预测并评估
rf_predictions = rf_model.predict(X_test_flat)
rf_rmse = mean_squared_error(FD003_test_y* 125, rf_predictions*125, squared=False)
print(f"随机森林 RMSE: {rf_rmse}")
print(f"Prediction completed in {end_time - start_time:.2f} seconds.")

# 模型融合：简单平均GPR和随机森林的预测
# ensemble_predictions = 0.5 * rf_predictions + 0.5 * mu.reshape(-1)
# ensemble_rmse = mean_squared_error(FD003_test_y, ensemble_predictions, squared=False)
# print(f"模型融合 RMSE: {ensemble_rmse}")

# 特征重要性
feature_importances = rf_model.feature_importances_
# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=[f"Feature {i}" for i in range(X_train_flat.shape[1])])
plt.title("Feature Importances")
plt.show()

# plt.figure()
# plt.scatter( [i for i in range(len(FD003_test_y))],FD003_test_y, c='red', label='Observations')
# plt.scatter( [i for i in range(len(mu))],mu, c='green', label='prediction')
# plt.legend()
# plt.savefig("/data/niutian/code/CMAPSS-release-master/FD003_1.png")
# plt.show()
