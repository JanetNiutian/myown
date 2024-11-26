from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

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

dir_path = '/data/niutian/code/CMAPSS-release-master/CMAPSSData/'
dataset = 'FD003'

scaler = StandardScaler()
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv((dir_path + 'train_{}.txt'.format(dataset)), sep='\s+', header=None, names=col_names)
test = pd.read_csv((dir_path + 'test_{}.txt'.format(dataset)), sep='\s+', header=None, names=col_names)


drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
drop_labels = setting_names + drop_sensors

train.drop(labels=drop_labels, axis=1, inplace=True)
test.drop(labels=drop_labels, axis=1, inplace=True)

# smooth_rate=40

# data_train = exponential_smoothing(train.values, smooth_rate)
# data_test = exponential_smoothing(test.values, smooth_rate)

# data_train = scaler.fit_transform(train) # (24720, 14)
# data_test = scaler.transform(test) # (16596, 14)

# print(test.shape)
# print(test)

from dtaidistance import dtw_ndim

def extract_multidimensional_series(df, engine_ids, sensors):
    """为指定engine编号提取多维时间序列数据"""
    series = {engine_id: df[df['unit_nr'] == engine_id][sensors].values for engine_id in engine_ids}
    return series


def dtw_ndim_distances(train_series, test_series, num_top=30):
    """计算DTW距离并返回每个测试engine的多个最相似的训练engines"""
    dtw_matrix = np.zeros((len(test_series), len(train_series)))
    train_keys = list(train_series.keys())
    for i, (test_key, test_seq) in enumerate(test_series.items()):
        for j, train_key in enumerate(train_keys):
            train_seq = train_series[train_key]
            distance = dtw_ndim.distance_fast(test_seq, train_seq)
            dtw_matrix[i][j] = distance
    top_indices = np.argsort(dtw_matrix, axis=1)[:, :num_top]
    return dtw_matrix, train_keys, top_indices

# 选择多个传感器数据进行分析
sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
specific_test_ids = [25, 26, 45, 55, 58, 62, 63, 70, 75, 93]
test_series = extract_multidimensional_series(test, specific_test_ids, sensors)
train_series = extract_multidimensional_series(train, train['unit_nr'].unique(), sensors)

distances, train_keys, top_indices = dtw_ndim_distances(train_series, test_series)

# 打印每个测试engine的相似的训练engines
# for test_idx, indices in enumerate(top_indices):
#     print(f"Test Engine {specific_test_ids[test_idx]} is similar to Train Engines:")
#     for idx in indices:
#         print(f"  Train Engine {train_keys[idx]} with DTW distance: {distances[test_idx, idx]}")
#     print("\n")


# 定义每个测试引擎相似的train引擎列表
similar_engines = {
    25: {23, 69, 56, 92, 100, 4, 44, 40, 14, 12, 63, 53, 31, 65, 99, 80, 67, 93, 58, 8, 50, 26, 15, 28, 70, 36, 79, 76, 90, 91},
    26: {53, 23, 14, 56, 92, 69, 76, 65, 50, 80, 40, 4, 93, 44, 100, 31, 12, 63, 36, 58, 79, 99, 67, 70, 91, 28, 26, 8, 90, 29},
    45: {70, 79, 12, 91, 80, 28, 90, 93, 50, 26, 99, 65, 58, 66, 67, 29, 100, 63, 40, 36, 14, 3, 35, 22, 23, 78, 15, 44, 83, 1},
    55: {23, 100, 69, 63, 12, 40, 56, 92, 14, 4, 44, 65, 99, 31, 53, 80, 67, 8, 93, 50, 26, 58, 36, 15, 70, 28, 79, 29, 90, 91},
    58: {12, 80, 40, 93, 65, 70, 99, 14, 58, 50, 44, 79, 23, 100, 67, 28, 63, 26, 36, 90, 91, 69, 29, 56, 15, 53, 4, 66, 3, 8},
    62: {26, 42, 3, 89, 35, 73, 11, 63, 60, 46, 12, 45, 100, 70, 90, 62, 83, 75, 66, 84, 33, 52, 65, 23, 2, 14, 28, 38, 17, 72},
    63: {70, 90, 91, 66, 79, 28, 50, 80, 12, 93, 26, 35, 29, 22, 3, 99, 58, 78, 65, 1, 67, 36, 83, 63, 100, 6, 40, 14, 15, 23},
    70: {92, 69, 23, 53, 56, 100, 31, 14, 44, 40, 4, 80, 12, 63, 65, 76, 99, 93, 50, 58, 67, 79, 8, 70, 26, 36, 28, 15, 91, 90},
    75: {26, 3, 35, 11, 70, 89, 46, 60, 90, 66, 42, 83, 45, 52, 73, 12, 63, 84, 62, 75, 33, 2, 72, 38, 28, 100, 21, 29, 22, 39},
    93: {70, 26, 90, 66, 3, 35, 28, 12, 79, 29, 91, 65, 50, 63, 80, 83, 22, 93, 99, 100, 14, 67, 6, 23, 1, 78, 58, 36, 46, 40}
}

# 计算所有测试引擎相似train引擎的并集和交集
all_similar_engines = set()
for engine_list in similar_engines.values():
    all_similar_engines.update(engine_list)  # 并集

common_similar_engines = set(all_similar_engines)  # 初始化交集为并集
for engine_list in similar_engines.values():
    common_similar_engines.intersection_update(engine_list)  # 交集

# print("Union of all similar Train Engines:", all_similar_engines) 
# {1, 2, 3, 4, 6, 8, 11, 12, 14, 15, 17, 21, 22, 23, 26, 28, 29, 31, 33, 35, 36, 38, 39, 40, 42, 44, 45, 46, 50, 52, 53, 56, 58, 60, 62, 63, 65, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 80, 83, 84, 89, 90, 91, 92, 93, 99, 100}
print(len(all_similar_engines))  # 57
# print("Intersection of all similar Train Engines:", common_similar_engines)
# {100, 90, 70, 12, 26, 28, 63}