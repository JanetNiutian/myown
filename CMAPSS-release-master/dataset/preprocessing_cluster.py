"""Loading data sets"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.cluster import KMeans


def add_rul_1(df):
    """
    :param df: raw data frame
    :return: data frame labeled with targets
    """
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]

    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def train_validate_split(valid_size, num_engine):
    indices = list(range(1, num_engine+1))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_engine))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


# data_preprocessing, exponential smoothing, cubic spline interpolation
def load_dataset(dir_path, dataset, max_rul, seq_len, use_exponential_smoothing, smooth_rate):
    """
    :param cut: upper limit for target RULs
    :return: grouped data per sample
    """
    # define filepath to read data

    # define column names for easy indexing
    scaler = StandardScaler()
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv((dir_path + 'train_{}.txt'.format(dataset)), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_{}.txt'.format(dataset)), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_{}.txt'.format(dataset)), sep='\s+', header=None, names=['RUL'])

    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    test.drop(labels=drop_labels, axis=1, inplace=True)
    new_column_names = train.columns
    title_test = test.iloc[:, 0:2]
    title_train = train.iloc[:, 0:2]
    if use_exponential_smoothing:
        data_train = exponential_smoothing(train.values, smooth_rate)
        data_test = exponential_smoothing(test.values, smooth_rate)
    else:
        data_train = train.iloc[:, 2:]
        data_test = test.iloc[:, 2:]
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    data_train = pd.DataFrame(data=np.c_[title_train, data_train], columns=new_column_names) # 创建了一个新的 DataFrame
    data_test = np.c_[title_test, data_test] # 简单地创建了一个 NumPy 数组

    data_train = add_rul_1(data_train)
    data_train['RUL'].clip(upper=max_rul, inplace=True)
    data_train['RUL'] = data_train['RUL'] / max_rul
    y_test['RUL'].clip(upper=max_rul, inplace=True)
    y_test['RUL'] = y_test['RUL'] / max_rul
    group = data_train.groupby(by="unit_nr")

    test_engine_id = np.unique(title_test.iloc[:, 0])
    interpolated_data_norm = []
    for i in test_engine_id:
        mask = (data_test[:, 0] == i)
        temp = data_test[mask, :]
        if len(temp) < seq_len:
            x = np.linspace(0, seq_len-1, len(temp))
            x_new = np.linspace(0, seq_len-1, seq_len)
            for j in range(2, temp.shape[1]):
                y = temp[:, j]
                tck = interpolate.splrep(x, y)
                y_new = interpolate.splev(x_new, tck)
                y_new = np.expand_dims(y_new, axis=1)
                if j == 2:
                    yy = y_new
                else:
                    yy = np.concatenate((yy, y_new), axis=1)
            engine_id_new = np.expand_dims(np.repeat(np.array(i), seq_len), axis=1)
            time_cycle_new = np.expand_dims(x_new + 1, 1)
            yy = np.concatenate((engine_id_new, time_cycle_new, yy), axis=1)
            interpolated_data_norm.append(yy)
        else:
            interpolated_data_norm.append(temp)

    interpolated_data_norm = np.concatenate(interpolated_data_norm, axis=0)
    interpolated_data_norm = pd.DataFrame(interpolated_data_norm, columns=new_column_names)
    group_test = interpolated_data_norm.groupby('unit_nr')

    return group, group_test, y_test


def gen_sequence(df, seq_length):
    data_array = df.to_numpy()
    num_units = data_array.shape[0]
    for start, stop in zip(range(0, num_units-seq_length+1), range(seq_length, num_units+1)):
        yield data_array[start:stop, :]


def gen_sequence_test_engine(df, seq_length):
    data_array = df.to_numpy()
    num_units = data_array.shape[0]
    temp_x = []
    for start, stop in zip(range(0, num_units-seq_length+1), range(seq_length, num_units+1)):
        temp_x.append(data_array[start:stop, :])
    temp_x = np.array(temp_x)
    return temp_x


def gen_sequence_test(df, seq_length, win_len, labels, engine_id):
    data_array = df.to_numpy()
    num_units = data_array.shape[0]
    temp_x = []

    for start, stop in zip(range(num_units-win_len-seq_length+1, num_units-seq_length+1),
                           range(num_units-win_len+1, num_units+1)):
        temp_x.append(data_array[start:stop, :])
    temp_x = np.array(temp_x)
    labels.reshape(-1)
    temp_y = [labels[engine_id-1] for i in range(win_len-1, -1, -1)]
    temp_y = np.array(temp_y)
    return temp_x, temp_y


def gen_labels(df, seq_length):
    data_array = df.to_numpy()
    num_units = data_array.shape[0]
    return data_array[seq_length-1:num_units+1]

def gen_cluster(input_array,num_clusters):
    # 将其变换成更易于处理的形状 (19072, 14, 30)
    array_transposed = np.transpose(input_array, (0, 2, 1))

    # 计算统计量
    mean_features = np.mean(array_transposed, axis=2)
    std_features = np.std(array_transposed, axis=2)
    min_features = np.min(array_transposed, axis=2)
    max_features = np.max(array_transposed, axis=2)

    # 将统计特征合并为一个特征数组
    cluster_features = np.concatenate([mean_features, std_features, min_features, max_features], axis=1)

    kmeans = KMeans(n_clusters = num_clusters, random_state=0)
    output_cluster = kmeans.fit_predict(cluster_features)  # 对统计特征进行聚类

    return output_cluster, kmeans

def organize_by_cluster(input_array, output_cluster):
    # 初始化字典来存储每个聚类的数据
    cluster_dict = {i: [] for i in np.unique(output_cluster)}

    # 遍历聚类结果，将原始数据分配到相应的聚类中
    for index, cluster_id in enumerate(output_cluster):
        cluster_dict[cluster_id].append(input_array[index])

    # 将列表转换为 NumPy 数组，方便后续处理
    for key in cluster_dict:
        cluster_dict[key] = np.array(cluster_dict[key])

    return cluster_dict

def organize_labels_by_cluster(labels, cluster_indices):
    """
    根据聚类结果重新组织标签数据。
    
    :param labels: 原始标签数组，形状为 (n_samples, )
    :param cluster_indices: 每个样本对应的聚类索引，形状为 (n_samples, )
    :return: 一个按聚类索引组织的标签列表
    """
    # 获取聚类的数量
    num_clusters = np.unique(cluster_indices).size
    # num_clusters = 10
    print(num_clusters) 
    # 初始化每个聚类的标签列表
    clustered_labels = [[] for _ in range(num_clusters)]
    
    # 遍历所有样本，根据其聚类索引添加到相应的聚类标签列表中
    for idx, label in zip(cluster_indices, labels):
        clustered_labels[idx].append(label)
    
    # 将列表中的元素转换为 numpy 数组
    clustered_labels = [np.array(labels) for labels in clustered_labels]
    return clustered_labels



def use_cluster(input_array,kmeans,original_indices):

    # 将其变换成更易于处理的形状 (19072, 14, 30)
    array_transposed = np.transpose(input_array, (0, 2, 1))

    # 计算统计量
    mean_features = np.mean(array_transposed, axis=2)
    std_features = np.std(array_transposed, axis=2)
    min_features = np.min(array_transposed, axis=2)
    max_features = np.max(array_transposed, axis=2)

    # 将统计特征合并为一个特征数组
    cluster_features = np.concatenate([mean_features, std_features, min_features, max_features], axis=1)

    output_cluster = kmeans.predict(cluster_features)  # 对统计特征进行聚类

    # 根据聚类结果组织数据和索引
    clustered_data = {}
    clustered_indices = {}
    for i in range(kmeans.n_clusters):
        cluster_mask = (output_cluster == i)
        clustered_data[i] = input_array[cluster_mask]
        clustered_indices[i] = original_indices[cluster_mask]
    return output_cluster,clustered_indices

def get_dataloader_1(dir_path, sub_dataset, max_rul, seq_length,
                   use_exponential_smoothing, smooth_rate, test_seq_length):
    np.random.seed(1)
    group_train, group_test, y_test = load_dataset(dir_path,
                                                   sub_dataset,
                                                   max_rul,
                                                   seq_length,
                                                   use_exponential_smoothing,
                                                   smooth_rate=smooth_rate)
    num_train, num_test = len(group_train.size()), len(group_test.size())
    train_idx, valid_index = train_validate_split(0.1, num_train)

    train_array = np.array([])
    train_labels = np.array([])
    valid_array = np.array([])
    valid_labels = np.array([])
    test_array = np.array([])
    test_labels = np.array([])
    test_array_last = np.array([])
    test_labels_last = np.array([])
    num_test_windows = []
    test_engine_id = np.random.randint(1, 101)

    for i, j in enumerate(train_idx):

        x, y = group_train.get_group(j).iloc[:, 2:-1], \
               group_train.get_group(j).iloc[:, -1]
        seq_train = np.concatenate(list(gen_sequence(x, seq_length=seq_length)), 0)
        seq_labels = gen_labels(y, seq_length=seq_length)
        seq_train = np.reshape(seq_train, (x.shape[0] - seq_length + 1, seq_length, x.shape[1]))
        seq_labels = np.reshape(seq_labels, (y.shape[0] - seq_length + 1, 1))

        if i == 0:
            train_array = seq_train
            train_labels = seq_labels
            # train_visualize = (torch.tensor(seq_train, dtype=torch.float32),
            #                    torch.tensor(seq_labels, dtype=torch.float32))
            # engine_id = j

        else:
            train_array = np.concatenate([train_array, seq_train])
            train_labels = np.concatenate([train_labels, seq_labels])

    for i, j in enumerate(valid_index):
        x, y = group_train.get_group(j).iloc[:, 2:-1], \
               group_train.get_group(j).iloc[:, -1]
        seq_valid = np.concatenate(list(gen_sequence(x, seq_length=seq_length)), 0)
        seq_labels = gen_labels(y, seq_length=seq_length)
        seq_valid = np.reshape(seq_valid, (x.shape[0] - seq_length + 1, seq_length, x.shape[1]))
        seq_labels = np.reshape(seq_labels, (y.shape[0] - seq_length + 1, 1))

        if i == 0:
            valid_array = seq_valid
            valid_labels = seq_labels
        else:
            valid_array = np.concatenate([valid_array, seq_valid])
            valid_labels = np.concatenate([valid_labels, seq_labels])

    for i in range(1, num_test + 1):
        x_test = group_test.get_group(i).iloc[:, 2:]
        x_test_len = x_test.shape[0]
        required_len = test_seq_length + seq_length - 1
        if required_len > x_test_len:
            seq_test, seq_labels = gen_sequence_test(x_test, seq_length, x_test_len-seq_length+1, y_test.values, i)
            num_test_windows.append(x_test_len-seq_length+1)
        else:
            seq_test, seq_labels = gen_sequence_test(x_test, seq_length, test_seq_length, y_test.values, i)
            num_test_windows.append(test_seq_length)
        if i == 1:
            test_array = seq_test
            test_labels = seq_labels
        else:
            test_array = np.concatenate([test_array, seq_test])
            test_labels = np.concatenate([test_labels, seq_labels])
        if i == test_engine_id:
            test_visualize = (torch.tensor(seq_test, dtype=torch.float32),
                              torch.tensor(seq_labels, dtype=torch.float32))

    # 只用测试集中每个引擎的最后一个样本
    for i in range(1, num_test + 1):
        x_test = group_test.get_group(i).iloc[:, 2:]
        x_test_len = x_test.shape[0]
        seq_test, seq_labels = gen_sequence_test(x_test, seq_length, 1, y_test.values, i)
        if i == 1:
            test_array_last = seq_test
            test_labels_last = seq_labels
        else:
            test_array_last = np.concatenate([test_array_last, seq_test])
            test_labels_last = np.concatenate([test_labels_last, seq_labels])

    # train_array 的形状是 (19072, 30, 14)
    n_samples = test_array.shape[0]
    original_indices_test = np.arange(n_samples) 

    n_samples = valid_array.shape[0]
    original_indices_valid = np.arange(n_samples) 

    n_samples = test_array_last.shape[0]
    original_indices_test_last = np.arange(n_samples) 

    train_output_cluster,kmeans = gen_cluster(train_array,5)
    valid_output_cluster,valid_cluster_indices = use_cluster(valid_array,kmeans,original_indices_valid)
    test_output_cluster,test_cluster_indices = use_cluster(test_array,kmeans,original_indices_test)
    test_output_cluster_last,test_last_cluster_indices = use_cluster(test_array_last,kmeans,original_indices_test_last)

    train_cluster = organize_by_cluster(train_array, train_output_cluster)
    valid_cluster = organize_by_cluster(valid_array, valid_output_cluster)
    test_cluster = organize_by_cluster(test_array, test_output_cluster)
    test_cluster_last = organize_by_cluster(test_array_last, test_output_cluster_last)

    train_labels_clustered = organize_labels_by_cluster(train_labels, train_output_cluster)
    valid_labels_clustered = organize_labels_by_cluster(valid_labels, valid_output_cluster)
    test_labels_clustered = organize_labels_by_cluster(test_labels, test_output_cluster)
    test_labels_clustered_last = organize_labels_by_cluster(test_labels_last, test_output_cluster_last)

    # print("Cluster 0 data shape:", train_cluster[0].shape) (7028, 30, 14)
    # print(train_labels.shape) (19072, 1)
    # print(train_labels_clustered[0].shape) (7028, 1)
    # print(train_labels_clustered[1].shape) # (4130, 1)

    return train_cluster, train_labels_clustered, valid_cluster, valid_labels_clustered, test_cluster, test_labels_clustered,  test_cluster_last, test_labels_clustered_last, num_test_windows,valid_cluster_indices, test_cluster_indices, test_last_cluster_indices


def get_trainloader_2(input_array, labels):

    loaders = []
    num_clusters = 5 

    # print("Labels structure at start:", type(labels), len(labels)) Labels structure at start: <class 'list'> 5
    for cluster in range(num_clusters):
        cluster_features = input_array[cluster]
        # (7028, 30, 14)(4130, 30, 14)(1974, 30, 14)(3577, 30, 14)(2363, 30, 14)(1531, 30, 14)(216, 30, 14)(577, 30, 14)(75, 30, 14)(349, 30, 14)
        # (168, 30, 14)(130, 30, 14)(88, 30, 14)(90, 30, 14)(24, 30, 14)
        cluster_labels = labels[cluster]

        # 为每个聚类创建一个 PyTorch Dataset

        cluster_dataset = CmapssDataSet(cluster_features, cluster_labels)
        cluster_loader = DataLoader(dataset=cluster_dataset,
                                    batch_size=128,
                                    shuffle=True,
                                    drop_last=False)

        loaders.append(cluster_loader)
    return loaders

def get_dataloader_2(input_array, labels):

    loaders = []
    num_clusters = 5 

    # print("Labels structure at start:", type(labels), len(labels)) Labels structure at start: <class 'list'> 5
    for cluster in range(num_clusters):
        cluster_features = input_array[cluster]
        # (7028, 30, 14)(4130, 30, 14)(1974, 30, 14)(3577, 30, 14)(2363, 30, 14)(1531, 30, 14)(216, 30, 14)(577, 30, 14)(75, 30, 14)(349, 30, 14)
        # (168, 30, 14)(130, 30, 14)(88, 30, 14)(90, 30, 14)(24, 30, 14)
        cluster_labels = labels[cluster]

        # 为每个聚类创建一个 PyTorch Dataset

        cluster_dataset = CmapssDataSet(cluster_features, cluster_labels)
        cluster_loader = DataLoader(dataset=cluster_dataset,
                                    batch_size=len(cluster_labels),
                                    shuffle=False,
                                    drop_last=False)

        loaders.append(cluster_loader)
    return loaders

def exponential_smoothing(data_group, s):
    alpha = 2 / (1 + s)
    engine_id = np.unique(data_group[:, 0])
    data_smoothed = np.array([])
    for i in engine_id:
        mask = (data_group[:, 0] == i)
        data = data_group[mask, 2:]
        weights = [(1 - alpha) ** i for i in range(data.shape[0])]
        weights = np.cumsum(weights).reshape(data.shape[0], 1)
        weights = np.repeat(weights, data.shape[1], axis=1)
        data_new = np.zeros((data.shape[0], data.shape[1]))
        for row in range(data.shape[0]):
            weight_temp = np.array([(1 - alpha) ** i for i in range(row, -1, -1)]).reshape(row+1, 1)  # 最后一个是step
            weight_temp = np.repeat(weight_temp, data.shape[1], axis=1)
            data_new[row, :] = np.sum(np.multiply(data[:row+1, :], weight_temp), axis=0)
        data_new = data_new / weights
        if i == 1:
            data_smoothed = data_new
        else:
            data_smoothed = np.concatenate((data_smoothed, data_new), axis=0)
    return data_smoothed


def get_test_engine(dir_path, sub_dataset, max_rul, seq_length, smooth_rate, engine_id):
    group_train, group_test, y_test = load_dataset(dir_path,
                                                   sub_dataset,
                                                   max_rul,
                                                   seq_length,
                                                   use_exponential_smoothing=True,
                                                   smooth_rate=smooth_rate)
    y_test *= 125
    test_engine_id = engine_id
    x_test = group_test.get_group(test_engine_id).iloc[:, 2:]
    x_test_len = len(x_test)
    test_array = gen_sequence_test_engine(x_test, seq_length=seq_length)
    test_label = y_test.iloc[test_engine_id-1].astype('int').item()
    test_label = pd.Series(list(range(x_test_len-seq_length+test_label, test_label-1, -1)))
    test_label.clip(upper=max_rul, inplace=True)
    test_label = test_label.values
    cmapssDataset_test_engine = CmapssDataSet(test_array, test_label)
    test_loader = DataLoader(dataset=cmapssDataset_test_engine,
                             batch_size=x_test_len,
                             shuffle=False)
    return test_loader


class CmapssDataSet(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32).clone().detach()
        self.y_data = torch.tensor(y_data, dtype=torch.float32).clone().detach()
        self.len = y_data.shape[0]
        # print(f"Initializing dataset with {self.len} samples.")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))