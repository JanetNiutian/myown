import pickle
import matplotlib.pyplot as plt

f_read = open('/data/niutian/code/CMAPSS-release-master/FD003_test_last_y.pkl', 'rb')
FD003_test_last_y = pickle.load(f_read)
# print(len(FD003_test_last_y))  # 100
# print(FD003_test_last_y.shape) # (100, 1) 
# print(FD003_test_last_y)

max_rul = 125

FD003_test_last_y_true = FD003_test_last_y*125
# print(FD003_test_last_y_true)

plt.figure(figsize=(10, 6))
plt.axvline(x=100, c='r', linestyle='--')
plt.plot(FD003_test_last_y_true, label='Actual Data')
# plt.plot(pred_rul, label='Predicted Data')
plt.title('RUL Prediction on FD003')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Remaining Useful Life")
plt.savefig('/data/niutian/code/CMAPSS-release-master/FD003_test_last.png')
# plt.show()