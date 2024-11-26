import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# def pic(distances,distances_test):
#     sns.set()
#     f, axs = plt.subplots(5, 3,figsize=(30,15))
#     axs = axs.ravel()
#     print(distances.shape[1])
#     for i in range(distances.shape[1]):
#         sns.histplot(distances[:,i], ax = axs[i],kde=True ,label="dimension :{}".format(i+1))
#         sns.histplot(distances_test[:,i], ax = axs[i],kde=True ,stat='probability',label="dimension :{}".format(i+1))
#         #axs[i].set_xlabel('Wasserstein Distance') 
#         #axs[i].set_title('Distribution ')
#         axs[i].legend()
#     plt.subplots_adjust(wspace=0.2, hspace=0.2)
#     plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_distribution.png')
#     plt.show()




# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
# FD003_train_x = pickle.load(f_read)

# length , sequence_len, feature_num = FD003_train_x.shape
# FD003_train_x_new = FD003_train_x.reshape(length * sequence_len , feature_num )


# # print(FD003_train_x_new.shape) # (19072, 30, 14) --sequence-len,--feature-num
# # pic(FD003_train_x_new)


# # plt.title('Distribution')

# # sns.histplot(data=FD003_train_x_new[:, 0],kde=True)

# # plt.tick_params(labelsize=30)

# # plt.show()







# # f_read = open('C:/Users/Admin/Desktop/pythoncode/博二上/timeseries/FD003_train_y.pkl', 'rb')
# # FD003_train_y = pickle.load(f_read)
# # print(FD003_train_y.shape)  # 19072


# f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_x.pkl', 'rb')
# FD003_test_x = pickle.load(f_read)
# print(FD003_test_x.shape) 
# length1 , sequence_len1, feature_num1 = FD003_test_x.shape
# FD003_test_x_new = FD003_test_x.reshape(length1 * sequence_len1 , feature_num1 )
# print(FD003_test_x_new.shape) 
# pic(FD003_train_x_new, FD003_test_x_new)
# f_read = open('C:/Users/Admin/Desktop/pythoncode/博二上/timeseries/FD003_test_last_y.pkl', 'rb')
# FD003_test_y = pickle.load(f_read)

# model  = GPR(FD003_train_x[:5000], FD003_train_y[:5000] , FD003_test_x , FD003_test_y)
# mu ,uncertainty= model.predict()
# plt.figure()
# plt.scatter( [i for i in range(len(FD003_test_y))],FD003_test_y, c='red', label='Observations')
# plt.plot( [i for i in range(len(mu))],mu, c='green', label='prediction')
# plt.fill_between([i for i in range(len(mu))], mu + uncertainty, mu - uncertainty, alpha=0.5)
# plt.legend()
# plt.savefig("C:/Users/Admin/Desktop/pythoncode/博二上/timeseries/1.png")
# plt.show()

f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD004_train_x.pkl', 'rb')
FD004_train_x = pickle.load(f_read)

f_read = open('/home/niutian/原data/code/CMAPSS-release-master/FD004_test_last_x.pkl', 'rb')
FD004_test_x = pickle.load(f_read)

# 将数据展平成二维数组，以便绘制直方图或密度图
flat_train_data = FD004_train_x.reshape(-1, 14)

flat_test_data = FD004_test_x.reshape(-1, 14)

# 获取特征名称，假设你的特征名称存在 feature_names 中
# 如果没有特征名称，可以自己定义
feature_names = [f"Feature {i+1}" for i in range(14)]
# 设置全局字体大小
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 7, figsize=(20, 8))  # 3行5列
axs = axs.flatten() 

# 绘制训练集数据
for i in range(14):
    sns.kdeplot(flat_train_data[:, i], ax=axs[i], label='train')
    sns.kdeplot(flat_test_data[:, i], ax=axs[i],  label='test', color='r', linestyle='--')    
    axs[i].set_title(f'Feature {i+1}', fontsize=14)
    axs[i].tick_params(axis='both', which='major', labelsize=14) 


# 移除多余的子图
for j in range(14, len(axs)):
    fig.delaxes(axs[j])



# 调整子图布局
plt.tight_layout()

# 添加图例（仅在最后一个子图中添加）
# axes[2, 4].legend()

plt.savefig('/home/niutian/原data/code/CMAPSS-release-master/FD004_distribution.pdf')
