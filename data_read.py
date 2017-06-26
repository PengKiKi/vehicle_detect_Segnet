import cv2
import glob
import cv2
import pickle

data_file = 'ProcessedData.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

x_train=data['X_train']
x_val=data['X_val']
x_test=data['X_test']
y_train=data['y_train']
y_val=data['y_val']
y_test=data['y_test']

print(len(x_val))

print(x_train[1])
cv2.imshow("test",x_train[1])
print(y_train[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

