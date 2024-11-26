import pickle

f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_x.pkl', 'rb')
FD003_train_x = pickle.load(f_read)
print(len(FD003_train_x))  # 19072
print(FD003_train_x.shape) # (19072, 30, 14) --sequence-len,--feature-num
print(FD003_train_x)

f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_train_y.pkl', 'rb')
FD003_train_y = pickle.load(f_read)
print(len(FD003_train_y))  # 19072
print(FD003_train_y.shape) # (19072, 1) 
print(FD003_train_y)

