import pickle
import numpy as np
import matplotlib.pyplot as plt 
import time
import torch
import os
from dataset import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from train_LSTM_GPR import *
from train_LSTM_RFR import *

train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

rfr_meta_predictions,rfr_meta_predictions_importances = RFR_test_average(valid_loader)  
gpr_meta_predictions,gpr_meta_predictions_std = GPR_test(valid_loader)  # GPR模型的预测

# 组合基模型的预测作为新特征
X_meta_features = np.column_stack((rfr_meta_predictions, gpr_meta_predictions))

labels_list = []
with torch.no_grad():
    for inputs, labels in valid_loader:
        labels_list.append(labels.numpy())
y_meta = np.concatenate(labels_list, axis=0)

labels_list = []
with torch.no_grad():
    for inputs, labels in test_loader_last:
        labels_list.append(labels.numpy())
test_labels = np.concatenate(labels_list, axis=0)

# 训练元模型
start_time = time.time()
meta_model = LinearRegression().fit(X_meta_features, y_meta)
end_time = time.time()
print(f"meta Prediction completed in {end_time - start_time:.2f} seconds.")

# 使用基模型在测试数据上做预测
rfr_test_predictions,rfr_test_predictions_importances = RFR_test_average(test_loader_last)  
gpr_test_predictions,gpr_test_predictions_std = GPR_test(test_loader_last)

# 组合基模型的测试预测作为新特征
X_test_features = np.column_stack((rfr_test_predictions, gpr_test_predictions))

# 元模型做最终预测
final_predictions = meta_model.predict(X_test_features)

# 评估模型性能
rmse = mean_squared_error(test_labels*125, final_predictions*125, squared=False)
print(f'Test RMSE: {rmse}')

final_predictions = final_predictions.flatten()  # 确保是一维
# print(gpr_std)
gpr_meta_predictions_std = gpr_meta_predictions_std.flatten()  # 确保是一维
test_labels = test_labels.flatten()  # 确保是一维

# 可视化预测结果和实际值
plt.figure(figsize=(12, 6))
plt.plot(test_labels*125, label='Actual RUL')
plt.plot(final_predictions*125, label='Predicted RUL')
# plt.fill_between(range(len(test_labels)), final_predictions*125 - gpr_meta_predictions_std*125, final_predictions*125 + gpr_meta_predictions_std*125, alpha=0.4)
plt.xlabel('Sample')
plt.ylabel('RUL')
plt.legend()
plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_meta.png')
plt.show()

plt.figure(figsize=(12, 6))
features = [f'Feature {i+1}' for i in range(14)] # + ['LSTM_Feature']
plt.barh(features, rfr_meta_predictions_importances[:-1])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_meta_Importance.png')






