import numpy as np
import matplotlib.pyplot as plt 
import time
import torch
from dataset import *
from sklearn.metrics import mean_squared_error
# from train_LSTM_RFR import *

class LSTMModel(nn.Module):
    def __init__(self, input_size=14, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # 取最后一个时间步的输出作为特征
        features = lstm_out[:, -1, :]
        predictions = self.linear(features)
        # 返回特征和预测结果
        return features, predictions

class EarlyStopping:
    """早停类"""
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        初始化早停类

        参数：
        - patience：在验证集上性能不再提升时，等待的 Epoch 数
        - delta：被认为是性能提升的最小变化量
        - verbose：是否打印出提前停止的信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        判断是否触发早停条件

        参数：
        - val_loss：验证集上的损失值

        返回值：
        - early_stop：True 表示触发了早停条件，否则为 False
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        if self.verbose:
            if self.early_stop:
                print("Early stopping")
        return self.early_stop

# 定义Huber损失
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        error = target - input
        is_small_error = torch.abs(error) < self.delta
        small_error_loss = 0.5 * torch.pow(error, 2)
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

# 实例化损失函数
huber_loss = HuberLoss(delta=1.0)

def LSTM_train(train_loader):
    # 创建早停对象
    early_stopping = EarlyStopping(patience=5, verbose=True)
    model_lstm = LSTMModel()
    model_lstm.train()
    train_features, train_labels = [], []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    # 训练循环
    n_epochs = 50  # 或者根据需要调整
    start_time = time.time()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:     
            optimizer.zero_grad()
            model_lstm.hidden_cell = (torch.zeros(1, inputs.size(0), model_lstm.hidden_layer_size),
                                torch.zeros(1, inputs.size(0), model_lstm.hidden_layer_size))

            features, predictions = model_lstm(inputs)
            train_features.append(features.detach().numpy())  # 保存特征
            train_labels.append(labels.numpy())
            # 从预测结果中提取出预测张量
            predictions = predictions.squeeze(1)
            loss = huber_loss(labels.squeeze(), predictions)
            # loss = criterion(predictions, labels.squeeze())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1} Loss: {loss.item()}')
        # 在验证集上评估模型
        model_lstm.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                # 前向传播
                _, outputs = model_lstm(inputs)
                outputs = outputs.squeeze(1)
                loss = huber_loss(labels.squeeze(), outputs)
                # loss = criterion(outputs, labels.squeeze())
                total_loss += loss.item()
                # 计算准确率等其他指标
                # 更新其他指标

        # 计算验证集上的平均损失等指标
        avg_loss = total_loss / len(valid_loader)
        # 记录并打印验证集上的指标
        print(f'Epoch [{epoch+1}/{n_epochs}], Validation Loss: {avg_loss}')

        # 判断是否触发早停策略
        # 如果验证集上的性能不再提高，可以停止训练
        if early_stopping(avg_loss):
            print("Validation performance hasn't improved. Early stopping...")
            break
    end_time = time.time()
    print(f"LSTM Prediction completed in {end_time - start_time:.2f} seconds.")
    return model_lstm


train_loader, valid_loader, a, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/home/niutian/原data/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD004',
            max_rul=125,
            seq_length=15,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

# test_egine1 = get_test_engine(dir_path='/home/xuzijun/CMAPSS-release-master/CMAPSSData/', sub_dataset='FD003', max_rul=125, seq_length=30, smooth_rate=128, engine_id=1)

model_lstm = LSTM_train(train_loader)


def extract_lstm_features(model, data_loader):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
            # LSTM模型处理输入数据
            lstm_features, _ = model(inputs)
            features.append(lstm_features.numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features, axis=0),np.concatenate(labels_list, axis=0)

def extract_lstm_features_test(model, x_test):  # 其实就是LSTM_test
    features = []
    model.eval()
    with torch.no_grad():
        # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
        # LSTM模型处理输入数据
        lstm_features, _ = model(x_test)
        features.append(lstm_features.numpy())
    return np.concatenate(features, axis=0)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C


def GPR_train(train_loader):

    # 定义高斯过程的核函数
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e3))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100, length_scale_bounds=(1e-1, 1e4))
    # 设置Matérn核，nu控制平滑性，length_scale控制相关性的距离范围
    # kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10), nu=1.5) # matern_
    # 设置Rational Quadratic核，alpha参数控制不同长度尺度的混合程度
    kernel = C(1.0, (1e-7, 1e3)) * RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-1, 10), alpha=1.0) # rq_  #-8太大了，尝试-6(15.82)，尝试-5(16.0)
    train_lstm_features,train_labels = extract_lstm_features(model_lstm, train_loader)
    # 创建高斯过程回归模型
    # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1.0)  # 增加n_restarts_optimizer和alpha
    # 随机生成索引
    total_samples = train_lstm_features.shape[0]
    indices = np.random.choice(total_samples, 5000, replace=False)
    # 使用LSTM特征和训练标签训练GPR模型
    start_time = time.time()
    gpr.fit(train_lstm_features[indices], train_labels[indices])
    end_time = time.time()
    print(f"GPR Prediction completed in {end_time - start_time:.2f} seconds.")

    return gpr
#write to txt file
def save_rmse_to_file(engine_id, rmse):
    with open('/home/niutian/原data/code/CMAPSS-release-master/LSTM_GPR_rmse_results.txt', 'a') as file:
        file.write(f'Engine ID: {engine_id}, RMSE: {rmse}\n')



def GPR_test(gpr,test_loader,engine_id):
 # gpr = GPR_train(train_loader)
    test_lstm_features,test_labels = extract_lstm_features(model_lstm, test_loader)
    # print(test_lstm_features.shape)  # (204,100)
    # print(test_lstm_features)
    # 使用测试特征进行预测
    gpr_predictions, gpr_std = gpr.predict(test_lstm_features, return_std=True)
    # print(gpr_predictions.shape)
    # print(gpr_predictions)
    gpr_predictions = gpr_predictions.flatten()  # 确保是一维
    # print(gpr_std)
    gpr_std = gpr_std.flatten()  # 确保是一维
    test_labels = test_labels.flatten()  # 确保是一维
    # 计算测试RMSE
    gpr_rmse = mean_squared_error(test_labels, gpr_predictions*125, squared=False)
    #write rmse result to file
    save_rmse_to_file(engine_id, gpr_rmse)
    print(f'GPR Test RMSE: {gpr_rmse}')
    # 时间轴（样本）
    # 时间轴（从第35个操作周期开始）
    time = np.arange(125, 125 + len(test_labels))
    # # 可视化预测结果和实际值
    # 设置图片的长宽比，增加高度，减少宽度
    plt.figure(figsize=(8, 12))  # 宽度为6，高度为12，比例较为"窄"
    plt.plot(time,test_labels, color='blue') # label='Actual RUL'
    # plt.plot(gpr_predictions*125, color='purple')
    plt.plot(time,gpr_predictions*125 - 1.96*gpr_std*125, color='green')
    plt.plot(time,gpr_predictions*125 + 1.96*gpr_std*125, color='red')
    plt.fill_between(time, gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125, alpha=0.3,color='pink' ) #,range(len(test_labels))
    # 计算误差
    errors = test_labels - gpr_predictions*125
    # 创建颜色数组，误差大于0时为绿色，小于0时为红色
    colors = ['green' if e > 0 else 'red' for e in errors]
    # 绘制误差条形图
    plt.bar(time, errors, color=colors, alpha=0.5)
    plt.xlabel('Operating Cycle', fontsize=16)
    plt.ylabel('RUL', fontsize=16)
    plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=16) 
    # plt.legend()
    plt.savefig(f'/home/niutian/原data/code/CMAPSS-release-master/single/FD004_{engine_id}_GPR.pdf')
    # plt.show()

    return gpr_predictions, gpr_std


if __name__ == '__main__':
    gpr = GPR_train(train_loader)
    for i in range(1, 248):
        if i == 32 or i == 68:
            engine_id = i
            test_egine = get_test_engine(dir_path='/home/niutian/原data/code/CMAPSS-release-master/CMAPSSData/', sub_dataset='FD004', max_rul=125, seq_length=15, smooth_rate=128, engine_id=i)
            GPR_test(gpr,test_egine,engine_id)
    # GPR_test_average(test_loader)
    # GPR_test(gpr,test_egine1)



