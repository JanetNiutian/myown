import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import os
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
        self.online_x = online_x
        self.online_y = online_y
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

input_dim = 420 # 30 * 14  # 因为seq_length=30，每个时间步的特征数量=14
encoding_dim = 100  # 假设的编码维度，可以根据需要调整，需要比input_dim小才能保证降维
# 设计自编码器架构
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid(),  # 使用Sigmoid或其他适当的激活函数
        )

    def forward(self, x):
        x = x.view(-1, 420)  # 确保输入被正确地扁平化
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 30, 14)  # 如果需要，将输出重新塑形回原始形状
        return encoded, decoded
    
# def get_train_loader():
train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

def save_checkpoint(state, filename="autoencoder_checkpoint.pth.tar"):
    """保存模型和训练状态的checkpoint"""
    torch.save(state, filename)

def train_autoencoder(model, train_loader, epochs=20):
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_time = time.time()  # 记录训练开始时间
    model.train()

    best_loss = float('inf')  # 假设最佳损失初始为无穷大

    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            inputs = data[0]  # 假设数据是(inputs, targets)形式
            optimizer.zero_grad()
            _, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    
        # 可以在每个epoch结束后保存checkpoint
        # 或者只在模型表现提升时保存，这里使用后者
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join('/data/niutian/code/CMAPSS-release-master/checkpoints/', f'autoencoder_epoch_{epoch+1}.pth.tar'))

    end_time = time.time()  # 记录训练结束时间
    training_time = end_time - start_time  # 计算训练所需总时间
    print(f"Training completed in {training_time:.2f} seconds.")

model_auto = Autoencoder(input_dim, encoding_dim)
train_autoencoder(model_auto, train_loader, epochs=20)     


def extract_features(model, data_loader):
    model.eval()  # 将模型设置为评估模式
    features = []
    with torch.no_grad():  # 不计算梯度，减少内存消耗
        for inputs, _ in data_loader:
            encoded, _ = model(inputs)
            features.append(encoded)
    return torch.cat(features, dim=0)

# 使用训练集提取特征
encoded_train_features = extract_features(model_auto, train_loader) # (n_samples, encoding_dim)
# 假设你还有test_loader
encoded_test_features = extract_features(model_auto, test_loader)

f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
FD003_train_x = pickle.load(f_read)# (19072, 30, 14) --sequence-len,--feature-num
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_y.pkl', 'rb')
FD003_train_y = pickle.load(f_read) # 19072
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_x.pkl', 'rb')
FD003_test_x = pickle.load(f_read)
f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_y.pkl', 'rb')
FD003_test_y = pickle.load(f_read)

# 假设 autoencoder 是你的自编码器模型，data_loader 是你的数据加载器
# encoded_features = []
# for inputs, _ in train_loader:
#     # inputs = inputs.to(device)  # 确保数据移动到了正确的设备
#     with torch.no_grad():
#         encoded, _ = autoencoder(inputs)
#         encoded_features.append(encoded.cpu().numpy())  # 移动到CPU并转换为numpy数组

# encoded_features = np.concatenate(encoded_features, axis=0)
print(encoded_train_features.shape) # [19072, 100]
print(encoded_train_features)
print(encoded_test_features.shape) # [500, 100]
print(encoded_test_features)
model  = GPR(encoded_train_features[:100], FD003_train_y[:100], encoded_test_features , FD003_test_y)
mu , s= model.predict()
print(mu.shape) # (500,1)
print(mu)
# save_model(model, '/data/niutian/code/CMAPSS-release-master/checkpoints/GPR_FD003_autoencoder.pkl')
# RMSE, score = evaluate(mu*125, FD003_test_y*125)
# print(RMSE, score)
# model_loaded = load_model('/data/niutian/code/CMAPSS-release-master/checkpoints/GPR_FD003_autoencoder.pkl')
# mu , s = model_loaded.predict()
# print(mu.shape)
# plt.figure()
# plt.scatter( [i for i in range(len(FD003_test_y))],FD003_test_y, c='red', label='Observations')
# plt.scatter( [i for i in range(len(mu))],mu, c='green', label='prediction')
# plt.legend()
# plt.savefig("/data/niutian/code/CMAPSS-release-master/FD003_1.png")
# plt.show()
