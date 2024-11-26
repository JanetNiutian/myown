import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
from dataset import *
from sklearn.metrics import mean_squared_error

class exp_kernel:
    def __init__(self ,length = 1,sigma_f = 1):
        self.length = length
        self.sigma_f = sigma_f
    def __call__(self , x1 ,x2):
        y = np.linalg.norm(x1-x2)
        return float(self.sigma_f*np.exp(-y**2 / (2*self.length*2)))
def K_matrix(x1,x2,kernel_function):
    return np.array([[kernel_function(a,b) for a in x1] for b in x2])

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

class GPR:
    def __init__(self ,data_x ,data_y ,online_x ,online_y ,kernel_function = exp_kernel(),noise = 1e-7 ):
        start_time = time.time()  # 开始时间
        print("Initializing GPR model...")
        self.data_x = data_x
        self.data_y = data_y
        self.online_x = online_x.numpy()  # 将 online_x 转换为 numpy 数组
        self.online_y = online_y.numpy()  # 将 online_y 转换为 numpy 数组
        self.kernel_function = kernel_function
        self.noise = noise
        print("Computing covariance matrix...")
        self.cov_matrix_inv = np.linalg.inv(K_matrix(data_x ,data_x , kernel_function) + (noise )* np.identity(len(data_x)))
        print("Model initialized.")
        end_time = time.time()  # 结束时间
        print(f"Initialization completed in {end_time - start_time:.2f} seconds.")
    def predict(self):
        start_time = time.time()  # 预测开始时间
        K_data_x = K_matrix(self.data_x , self.online_x ,self.kernel_function)  # m ,n
        K_x_x = K_matrix(self.online_x , self.online_x ,self.kernel_function)   # m ,m 
        mean = (K_data_x @ self.cov_matrix_inv @ self.data_y) 
        matrix = K_x_x - K_data_x @ self.cov_matrix_inv @  K_data_x.T
        s = np.sqrt(np.diag(matrix))
        s = s.reshape(-1, 1)
        end_time = time.time()  # 预测结束时间
        print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
        return mean, s

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            "data_x": model.data_x,
            "data_y": model.data_y,
            "cov_matrix_inv": model.cov_matrix_inv,
            "kernel_params": {"length": model.kernel_function.length, "sigma_f": model.kernel_function.sigma_f},
            "noise": model.noise
        }, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        kernel = exp_kernel(length=data["kernel_params"]["length"], sigma_f=data["kernel_params"]["sigma_f"])
        model = GPR(data_x=data["data_x"], data_y=data["data_y"], online_x=np.empty((0,)), online_y=np.empty((0,)), kernel_function=kernel, noise=data["noise"])
        model.cov_matrix_inv = data["cov_matrix_inv"]
    return model

def evaluate(y_test, true_pred):
    RMSE = mean_squared_error(y_test, true_pred, squared=False).item()
    score = compute_s_score(y_test, true_pred) # mean_pred_for_each_engine
    return RMSE, score

def calculate_performance_metrics(y_true, y_pred_lower, y_pred_upper):
    n = len(y_true)
    coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    range_y = np.max(y_true) - np.min(y_true)
    width_normalized = np.mean(y_pred_upper - y_pred_lower) / range_y
    alpha = 0.05
    cwc = width_normalized * np.exp((1 - coverage) / alpha)
    print(f'Coverage Probability: {coverage:.2f}')
    print(f'Normalized Average Width: {width_normalized:.2f}')
    print(f'Coverage Width-Based Criterion: {cwc:.2f}')
train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)
# train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
#             dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
#             sub_dataset='FD001',
#             max_rul=125,
#             seq_length=25,
#             batch_size=128,
#             use_exponential_smoothing=True,
#             smooth_rate=40)
# train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
#             dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
#             sub_dataset='FD002',
#             max_rul=125,
#             seq_length=20,
#             batch_size=128,
#             use_exponential_smoothing=True,
#             smooth_rate=40)
# train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
#             dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
#             sub_dataset='FD004',
#             max_rul=125,
#             seq_length=15,
#             batch_size=128,
#             use_exponential_smoothing=True,
#             smooth_rate=40)

features = []
labels_list = []

for inputs, labels in train_loader:
    # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
    features.append(inputs.numpy())
    labels_list.append(labels.numpy())

train_input = np.concatenate(features, axis=0)
train_labels = np.concatenate(labels_list, axis=0)
x_test, y_test = next(iter(test_loader)) # torch.Size([500, 30, 14]) y_test torch.Size([500, 1])
y_test = (y_test.reshape(-1)) * 125

start_time = time.time()
model  = GPR(train_input, train_labels, x_test , y_test)
end_time = time.time()
print(f"GPR Prediction completed in {end_time - start_time:.2f} seconds.")

rul_pred , gpr_std= model.predict()

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

mean_std_for_each_engine = [item.sum() / len(item) for item in std_for_each_engine]
mean_std_for_each_engine = np.take(mean_std_for_each_engine, index, axis=0)
mean_std_for_each_engine = np.floor(mean_std_for_each_engine) # 向下取整
calculate_performance_metrics(y_test, mean_pred_for_each_engine - 1.96*mean_std_for_each_engine, mean_pred_for_each_engine + 1.96*mean_std_for_each_engine)


RMSE, score = evaluate(y_test, mean_pred_for_each_engine)
print(f'Test RMSE: {RMSE}')
print(f'Test score: {score}')

# plt.figure(figsize=(12, 6))
# plt.plot(y_test, label='Actual RUL')
# plt.plot(mean_pred_for_each_engine, label='Predicted RUL')
# plt.xlabel('Sample')
# plt.ylabel('RUL')
# plt.legend()
# plt.savefig('/data/niutian/code/CMAPSS-release-master/FD002_GPR.png')
# plt.show()

# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
# FD003_train_x = pickle.load(f_read)# (19072, 30, 14) --sequence-len,--feature-num
# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_y.pkl', 'rb')
# FD003_train_y = pickle.load(f_read) # 19072
# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_x.pkl', 'rb')
# FD003_test_x = pickle.load(f_read)
# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_y.pkl', 'rb')
# FD003_test_y = pickle.load(f_read)

# model  = GPR(FD003_train_x, FD003_train_y, FD003_test_x , FD003_test_y)




