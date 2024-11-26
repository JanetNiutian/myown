from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataset.preprocessing_dtw import *
from torch import nn
import time
import torch
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

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
            for inputs, labels in valid_loaders[0]:
                # 前向传播
                _, outputs = model_lstm(inputs)
                outputs = outputs.squeeze(1)
                loss = huber_loss(labels.squeeze(), outputs)
                # loss = criterion(outputs, labels.squeeze())
                total_loss += loss.item()
                # 计算准确率等其他指标
                # 更新其他指标
        
        # 计算验证集上的平均损失等指标
        avg_loss = total_loss / len(valid_loaders[0])
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

train_cluster, train_labels_clustered, valid_cluster, valid_labels_clustered, test_cluster, test_labels_clustered,  num_test_windows,valid_cluster_indices, test_cluster_indices,num_clusters = get_dataloader_1(
            dir_path='/data/niutian/code/CMAPSS-release-master/CMAPSSData/',
            sub_dataset='FD003',
            max_rul=125,
            seq_length=30,
            use_exponential_smoothing=True,
            smooth_rate=40,
            test_seq_length=5)

train_loaders = get_trainloader_2(train_cluster, train_labels_clustered, num_clusters)
valid_loaders = get_dataloader_2(valid_cluster, valid_labels_clustered)
test_loaders = get_dataloader_2(test_cluster, test_labels_clustered)

cluster_models = []

for loader in train_loaders:
    # for inputs, labels in loader:
    #     print("Inputs shape:", inputs.shape)  # 打印输入数据的形状
    #     print("Labels shape:", labels.shape)  # 打印标签数据的形状
    #     break  # 只查看第一个批次，然后停止迭代
    model = LSTMModel()  # 每个聚类一个新的模型实例
    trained_model = LSTM_train(loader)  # 训练模型
    cluster_models.append(trained_model)

# for i, loader in enumerate(test_loaders):
#     print(f"Evaluating model for cluster {i}")
    # 这里使用 cluster_models[i] 来评估或测试
    # 比如: test_results = evaluate_model(cluster_models[i], loader)

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



def GPR_train(train_loader,model_lstm):

    # 定义高斯过程的核函数
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e3))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100, length_scale_bounds=(1e-1, 1e4))
    # 设置Matérn核，nu控制平滑性，length_scale控制相关性的距离范围
    # kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10), nu=1.5) # matern_
    # 设置Rational Quadratic核，alpha参数控制不同长度尺度的混合程度
    kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-1, 10)) # rq_
    train_lstm_features,train_labels = extract_lstm_features(model_lstm, train_loader)
    # 创建高斯过程回归模型
    # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1.0)  # 增加n_restarts_optimizer和alpha

    # 使用LSTM特征和训练标签训练GPR模型
    start_time = time.time()
    gpr.fit(train_lstm_features, train_labels)
    end_time = time.time()
    print(f"GPR Prediction completed in {end_time - start_time:.2f} seconds.")

    return gpr




def GPR_test_average():

    
    total_rul_pred = np.zeros(500)
    total_gpr_std = np.zeros(500)
    total_true = np.zeros(500)
    for i in range(num_clusters):
        model_lstm = cluster_models[i]
        gpr = GPR_train(train_loaders[i],model_lstm)
        x_test, y_test = next(iter(test_loaders[i]))
        y_test = (y_test.reshape(-1)) * 125
        test_lstm_features = extract_lstm_features_test(model_lstm, x_test)
        rul_pred, gpr_std = gpr.predict(test_lstm_features, return_std=True)
        indices = test_cluster_indices[i]
        total_rul_pred[indices] = rul_pred.reshape(-1)
        total_gpr_std[indices] = gpr_std.reshape(-1)
        total_true[indices] = y_test
        # print(y_test)
        # print(y_test.shape)

    
    total_rul_pred = total_rul_pred.reshape(-1)  # (497,1) -- (497,)
    total_rul_pred *= 125
    preds_for_each_engine = np.array_split(total_rul_pred, len(num_test_windows)) # list
    # print("reds_for_each_engine1",preds_for_each_engine)
    total_gpr_std = total_gpr_std.reshape(-1)  # (497,1) -- (497,)
    total_gpr_std *= 125
    std_for_each_engine = np.array_split(total_gpr_std, len(num_test_windows))

    # print(total_true)
    y_test_windows = np.array_split(total_true, len(num_test_windows))
    y_test_means = [item.mean() for item in y_test_windows]
    index = np.argsort(-np.array(y_test_means))
    total_true = np.array(y_test_means)[index]
    # print(total_true)
    mean_pred_for_each_engine = [item.sum() / len(item) for item in preds_for_each_engine]
    mean_pred_for_each_engine = np.take(mean_pred_for_each_engine, index, axis=0)
    mean_pred_for_each_engine = np.floor(mean_pred_for_each_engine) # 向下取整
    # print("mean_pred_for_each_engine",mean_pred_for_each_engine)
    mean_std_for_each_engine = [item.sum() / len(item) for item in std_for_each_engine]
    mean_std_for_each_engine = np.take(mean_std_for_each_engine, index, axis=0)
    mean_std_for_each_engine = np.floor(mean_std_for_each_engine) # 向下取整
    # 使用测试特征进行预测


    rf_rmse = mean_squared_error(total_true, mean_pred_for_each_engine, squared=False).item()
    print(f'Test RMSE: {rf_rmse}')

    # Calculate differences
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

    plt.figure(figsize=(12, 6))
    plt.plot(total_true, label='Actual RUL')
    plt.plot(mean_pred_for_each_engine, label='Predicted RUL')
    plt.fill_between(range(len(total_true)), mean_pred_for_each_engine - 1.96*mean_std_for_each_engine, mean_pred_for_each_engine + 1.96*mean_std_for_each_engine, alpha=0.4)
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_cluster.png')
    plt.show()

if __name__ == '__main__':
    GPR_test_average()