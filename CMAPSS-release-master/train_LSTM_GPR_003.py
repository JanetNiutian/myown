import numpy as np
import matplotlib.pyplot as plt 
import time
import torch
from dataset import *
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from train_LSTM_RFR import *

class LSTMModel(nn.Module):
    def __init__(self, input_size=14, hidden_layer_size=100, output_size=1, num_layers=3, dropout_rate=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
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
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
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
        scheduler.step(avg_loss) 
        # 判断是否触发早停策略
        # 如果验证集上的性能不再提高，可以停止训练
        if early_stopping(avg_loss):
            print("Validation performance hasn't improved. Early stopping...")
            break
    end_time = time.time()
    print(f"LSTM Prediction completed in {end_time - start_time:.2f} seconds.")
    return model_lstm


train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path='/home/niutian/原data/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            batch_size=128,
            use_exponential_smoothing=True,
            smooth_rate=40)

test_egine1 = get_test_engine(dir_path='/home/niutian/原data/code/CMAPSS-release-master/CMAPSSData/', sub_dataset='FD003', max_rul=125, seq_length=30, smooth_rate=128, engine_id=1)

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



# print(train_lstm_features) #[[-0.05101622  0.15596637 -0.3830421  ...  0.83650297  0.7793191
#    0.36806738]
#  ...
#  [ 0.28677619  0.17582004  0.14085098 ... -0.4133841  -0.45184755
#    0.66260266]]
# print(train_lstm_features.shape) # (19072, 100)
# print(train_labels) # [[0.44 ]
#  [0.56 ]
#  [0.92 ]
#  ...
#  [1.   ]
#  [0.944]
#  [1.   ]]
# print(train_labels.shape) # (19072, 1)
# print(test_lstm_features) 
# print(test_lstm_features.shape) # (100, 100)
# print(test_lstm_features)
# print(test_lstm_features.shape)


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
    kernel = C(1.0, (1e-7, 1e3)) * RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-1, 10), alpha=1.0) # rq_  #-8太大了，-7(15.29) 尝试-6(15.82)，尝试-5(16.0)
    train_lstm_features,train_labels = extract_lstm_features(model_lstm, train_loader)
    # 创建高斯过程回归模型
    # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1.0)  # 增加n_restarts_optimizer和alpha
    # 随机生成索引
    total_samples = train_lstm_features.shape[0]
    indices = np.random.choice(total_samples, 5250, replace=False)
    # 使用LSTM特征和训练标签训练GPR模型
    start_time = time.time()
    gpr.fit(train_lstm_features[indices], train_labels[indices])
    end_time = time.time()
    print(f"GPR Prediction completed in {end_time - start_time:.2f} seconds.")

    return gpr

def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    rul_true = torch.from_numpy(rul_true).float()
    rul_pred = torch.from_numpy(rul_pred).float()
    diff = rul_pred - rul_true
    return torch.sum(torch.where(diff < 0, torch.exp(-diff/13)-1, torch.exp(diff/10)-1))

def GPR_test_average(test_loader):

    # gpr = GPR_train(train_loader)
    x_test, y_test = next(iter(test_loader)) # torch.Size([500, 30, 14]) y_test torch.Size([500, 1])
    y_test = (y_test.reshape(-1)) * 125
    test_lstm_features = extract_lstm_features_test(model_lstm, x_test)
    # print(test_lstm_features.size)
    rul_pred, gpr_std = gpr.predict(test_lstm_features, return_std=True) # (500, 1) [[0.55744501][0.54677675][0.53615835][0.52466257][0.51270392][0.7166863 ][0.70877164]
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
    print(f'Score: {score}')
    # # Calculate differences
    # differences = y_test - mean_pred_for_each_engine
    # # Define a threshold for large differences
    # threshold = 30  # You can adjust this value based on your specific requirements
    # engine = []
    # # Print large differences
    # print("Large differences between actual and predicted RUL:")
    # for i, diff in enumerate(differences):
    #     if abs(diff) > threshold:
    #         print(f"Engine {i}: Actual RUL = {y_test[i]}, Predicted RUL = {mean_pred_for_each_engine[i]}, Difference = {diff}")
    #         engine.append(i)

    calculate_performance_metrics(y_test, mean_pred_for_each_engine - 1.96*mean_std_for_each_engine, mean_pred_for_each_engine + 1.96*mean_std_for_each_engine)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual RUL')
    plt.plot(mean_pred_for_each_engine, label='Predicted RUL')
    plt.fill_between(range(len(y_test)), mean_pred_for_each_engine - 1.96*mean_std_for_each_engine, mean_pred_for_each_engine + 1.96*mean_std_for_each_engine, alpha=0.4)
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD003_LSTM+GPR.png')
    plt.show()

    # 调用此函数展示分布
    # 假设`engine`和`differences`是前面代码中收集的引擎索引和它们的差值
    # plot_large_diff_distribution(engine, [test_lstm_features[i] for i in engine])

gpr = GPR_train(train_loader)

def GPR_test_single(test_loader): # single engine 
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
    print(f'GPR Test RMSE: {gpr_rmse}')
    calculate_performance_metrics(test_labels, gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125)
    # # 可视化预测结果和实际值
    plt.figure(figsize=(12, 6))
    plt.plot(test_labels, label='Actual RUL')
    plt.plot(gpr_predictions*125, label='Predicted RUL')
    plt.fill_between(range(len(test_labels)), gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125, alpha=0.4)
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD003_engine_1_GPR.png')
    # plt.show()

    return gpr_predictions,gpr_std

def GPR_test_last(test_loader): # 乱序last 
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
    gpr_rmse = mean_squared_error(test_labels*125, gpr_predictions*125, squared=False)
    score = compute_s_score(test_labels*125, gpr_predictions*125)
    print(f'GPR Test RMSE: {gpr_rmse}')
    print(f'Score: {score}')
    calculate_performance_metrics(test_labels*125, gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125)
    samples = np.arange(100)
    # # 可视化预测结果和实际值
    plt.figure(figsize=(12, 6))
    plt.plot(samples, test_labels*125, color='orange', label='Actual RUL')
    plt.scatter(samples, gpr_predictions*125, color='blue', label='Predicted RUL')
    # plt.plot(samples, gpr_predictions*125, 'b--', label='Predicted RUL')
    plt.fill_between(samples, gpr_predictions*125 - 1.96*gpr_std*125, gpr_predictions*125 + 1.96*gpr_std*125, color='purple', alpha=0.15, label='RUL PI estimation')
    plt.xlabel('Test case ID')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD003_last_5250.png')
    plt.plot(samples, gpr_predictions*125, 'b--', label='Predicted RUL')
    plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD003_last_lines_5250.png')
    
    # plt.show()

    return gpr_predictions,gpr_std


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



if __name__ == '__main__':
    GPR_test_average(test_loader)
    GPR_test_last(test_loader_last)
    # GPR_test(test_egine1)




