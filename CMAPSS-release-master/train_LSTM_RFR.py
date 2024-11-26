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
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error


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

# def get_train_loader():
train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/home/niutian/原data/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD004',
            max_rul=125,
            seq_length=15,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

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


def combine_features_train(model, data_loader):  # 其实就是LSTM_test
    combined_features = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
            # 我们取最后一个时间步的特征作为代表
            last_time_step_features = inputs[:, -1, :]
            # print(inputs.shape) # torch.Size([128, 30, 14])
            # 获取LSTM特征
            _, lstm_predictions = model(inputs)
            # 将LSTM特征和最后一个时间步的特征组合
            combined = torch.cat((last_time_step_features, lstm_predictions), dim=1)
            
            combined_features.append(combined.numpy())
            labels_list.append(labels.squeeze().numpy())

    # 将列表转换为Numpy数组
    combined_features = np.concatenate(combined_features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    return combined_features, labels_list

def combine_features_test(model, x_test):  # 其实就是LSTM_test
    combined_features = []
    model.eval()
    with torch.no_grad():
        # 假设 inputs 的形状为 [batch_size, seq_length, num_features]
        # 我们取最后一个时间步的特征作为代表
        last_time_step_features = x_test[:, -1, :]
            
        # 获取LSTM特征
        _, lstm_predictions = model(x_test)
        # 将LSTM特征和最后一个时间步的特征组合
        combined = torch.cat((last_time_step_features, lstm_predictions), dim=1)
            
        combined_features.append(combined.numpy())

    # 将列表转换为Numpy数组
    combined_features = np.concatenate(combined_features, axis=0)
    
    return combined_features


# print(train_combined_features) #[[-0.05101622  0.15596637 -0.3830421  ...  0.83650297  0.7793191
#    0.36806738]
#  [-0.43436238 -0.06163168 -0.34040684 ...  0.88187665  0.9215579
#    0.73038673]
#  [-0.40023175 -0.45768324 -0.47826797 ...  0.33122447  0.55430776
#    0.863125  ]
#  ...
#  [-0.42971838 -0.30951387 -0.5063626  ... -0.10129406 -0.00438201
#    0.9569867 ]
#  [ 1.2377148   0.84894454  1.2382845  ... -1.221891   -1.2061241
#    0.8190743 ]
#  [ 0.28677619  0.17582004  0.14085098 ... -0.4133841  -0.45184755
#    0.66260266]]
# print(train_combined_features.shape) # (19072, 15)
# print(train_labels) # [[0.44 ]
#  [0.56 ]
#  [0.92 ]
#  ...
#  [1.   ]
#  [0.944]
#  [1.   ]]
# print(train_labels.shape) # (19072, 1)
# print(test_combined_features) 
# print(test_combined_features.shape) # (500, 15)
# print(test_labels)
# print(test_labels.shape)


from sklearn.ensemble import RandomForestRegressor
model_lstm = LSTM_train(train_loader)

def RFR_train(train_loader):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_combined_features, train_labels = combine_features_train(model_lstm, train_loader)
    start_time = time.time()
    rf_model.fit(train_combined_features, train_labels.ravel())
    end_time = time.time()
    print(f"RFR Prediction completed in {end_time - start_time:.2f} seconds.")

    return rf_model

def RFR_test_average(test_loader):
    # 使用组合特征进行预测
    rf_model = RFR_train(train_loader)
    x_test, y_test = next(iter(test_loader))
    # print(x_test.shape) # torch.Size([500, 30, 14])
    y_test = (y_test.reshape(-1)) * 125
    # print(y_test.shape) torch.Size([500])
    # print("y_test:", y_test)  tensor([ 44.,  44.,  
    test_combined_features = combine_features_test(model_lstm, x_test)
    
    # print(test_combined_features.shape) # (500, 15)
    # print(test_combined_features)

    rul_pred = rf_model.predict(test_combined_features)
    rul_pred = rul_pred.reshape(-1)  # (497,1) -- (497,)
    rul_pred *= 125
    preds_for_each_engine = np.array_split(rul_pred, len(num_test_windows))
    # 将 y_test 分割成 num_test_windows 个子数组，并计算每个子数组的平均值
    y_test_windows = np.array_split(y_test, len(num_test_windows))
    y_test_means = [item.mean() for item in y_test_windows]
   
    # 使用 argsort() 方法获取排序后的索引
    index = np.argsort(-np.array(y_test_means))  # 使用负号表示降序排序

    # 根据排序后的索引重新排列 y_test_means
    y_test = np.array(y_test_means)[index]
    mean_pred_for_each_engine = [item.sum() / len(item) for item in preds_for_each_engine]
    mean_pred_for_each_engine = np.take(mean_pred_for_each_engine, index, axis=0)
    mean_pred_for_each_engine = np.floor(mean_pred_for_each_engine) # 向下取整
    
    # print(y_test.shape) (101,)
    # print("y_test:", y_test) [74.06061 44.           nan 
    # print(mean_pred_for_each_engine.shape) (101,)
    # print("mean_pred_for_each_engine:", mean_pred_for_each_engine) [77. 56. nan 

    # rf_rmse = mean_squared_error(y_test, mean_pred_for_each_engine, squared=False).item()
    # print(f'Test RMSE: {rf_rmse}')
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test, label='Actual RUL')
    # plt.plot(mean_pred_for_each_engine, label='Predicted RUL')
    # plt.xlabel('Sample')
    # plt.ylabel('RUL')
    # plt.legend()
    # plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD001_LSTM+RFR.png')
    # plt.show()


    # 特征重要性
    feature_importances = rf_model.feature_importances_
    feature_importances0 = feature_importances[:-1]
    normalized_importances = feature_importances0 / feature_importances0.sum()
    # 步骤2：对特征重要性和特征名称进行排序
    features = [f'Feature {i+1}' for i in range(14)]
    sorted_indices = np.argsort(-normalized_importances)
    sorted_importances = normalized_importances[sorted_indices]
    sorted_features = np.array(features)[sorted_indices]
    # # 步骤3：创建渐变颜色
    # cmap = plt.get_cmap('viridis')  # 使用 Viridis 色图作为示例
    # normalize = mcolors.Normalize(vmin=sorted_importances.min(), vmax=sorted_importances.max())
    # colors = cmap(normalize(sorted_importances))
    # 可视化特征重要性
    plt.figure(figsize=(16, 6))  # 增加图片的宽度，这里设置宽度为16，高度为6
    plt.bar(sorted_features, sorted_importances)  # 使用 bar 绘制图形

    # 设置字体大小
    plt.ylabel('Feature Importance', fontsize=16)  # 纵轴标签字体大小为16
    plt.xlabel('Feature', fontsize=16)  # 横轴标签字体大小为16
    plt.xticks(fontsize=12)  # 设置x轴标签的字体大小为14，并旋转45度
    plt.yticks(fontsize=14)  # 设置y轴刻度的字体大小为14

    # 自动调整布局，避免标签重叠
    plt.tight_layout()

    # 保存图像
    plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD004_Importance.pdf')

    return feature_importances

def RFR_test_last(test_loader_last):
    rf_model = RFR_train(train_loader)
    test_lstm_features,test_labels = combine_features_train(model_lstm, test_loader_last)
    # 使用测试特征进行预测
    rf_predictions = rf_model.predict(test_lstm_features)

    rf_predictions = rf_predictions.flatten()  # 确保是一维
    test_labels = test_labels.flatten()  # 确保是一维
    # 计算测试RMSE
    rf_rmse = mean_squared_error(test_labels*125, rf_predictions*125, squared=False)

    print(f'Test RMSE: {rf_rmse}')


if __name__ == '__main__':

    RFR_test_average(test_loader)










