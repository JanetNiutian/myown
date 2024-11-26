import pickle

# 读取 `train_cluster.pkl`
with open('/data/niutian/code/CMAPSS-release-master/cluster_middle/train_cluster.pkl', 'rb') as f:
    train_cluster = pickle.load(f)
    print("train_cluster:", train_cluster)

with open('/data/niutian/code/CMAPSS-release-master/cluster_middle/valid_cluster.pkl', 'rb') as f:
    valid_cluster = pickle.load(f)
    print("valid_cluster:", valid_cluster)

with open('/data/niutian/code/CMAPSS-release-master/cluster_middle/test_cluster.pkl', 'rb') as f:
    test_cluster = pickle.load(f)
    print("test_cluster:", test_cluster)

