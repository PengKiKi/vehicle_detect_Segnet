import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import sklearn.svm as svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib


# Load the training validation and test data
data_file = 'data.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
cars_train = data['cars_train']
cars_val = data['cars_val']
cars_test = data['cars_test']
notcars_train = data['notcars_train']
notcars_val = data['notcars_val']
notcars_test = data['notcars_test']

def single_img_features(img, color_space="HLS",channel=1):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    #cv2.imshow("test",feature_image)
    #cv2.waitKey(0)
    return feature_image[:,:,channel]


def get_features(files, color_space='HLS', channel=1):
    features = []
    for file in files:
        img = mpimg.imread(file)
        img_features = single_img_features(img, color_space=color_space, channel=channel)
        features.append(img_features)

    return features


color_space = 'HLS'  # RGB, HSV, LUV, HLS, YUV, YCrCb

#hog_channel = 1
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
cars_train_feat = get_features(cars_train, color_space)
cars_val_feat = get_features(cars_val, color_space)
cars_test_feat = get_features(cars_test, color_space)

notcars_train_feat = get_features(notcars_train, color_space)
notcars_val_feat = get_features(notcars_val, color_space)
notcars_test_feat = get_features(notcars_test, color_space)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HLS channel "L"')


cars_ntrain = len(cars_train_feat)
cars_nval = len(cars_val_feat)
cars_ntest = len(cars_test_feat)
ncars_ntrain = len(notcars_train_feat)
ncars_nval = len(notcars_val_feat)
ncars_ntest = len(notcars_test_feat)


y_train = np.hstack((np.ones(cars_ntrain), np.zeros(ncars_ntrain)))
y_val = np.hstack((np.ones(cars_nval), np.zeros(ncars_nval)))
y_test = np.hstack((np.ones(cars_ntest), np.zeros(ncars_ntest)))




X_train = cars_train_feat + notcars_train_feat
X_val = cars_val_feat + notcars_val_feat
X_test = cars_test_feat + notcars_test_feat
print(len(X_train))

X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC

font_size = 15
f, axarr = plt.subplots(4, 2, figsize=(20, 10))
f.subplots_adjust(hspace=0.2, wspace=0.05)

colorspace = cv2.COLOR_RGB2HLS
# colorspace=cv2.COLOR_RGB2HSV
# colorspace=cv2.COLOR_RGB2YCrCb

i1, i2 = 1, 2000

for ind, j in enumerate([i1, i2]):
    image = plt.imread(cars_train[j])
    feature_image = cv2.cvtColor(image, colorspace)

    axarr[ind, 0].imshow(image)
    axarr[ind, 0].set_xticks([])
    axarr[ind, 0].set_yticks([])
    title = "car {0}".format(j)
    axarr[ind, 0].set_title(title, fontsize=font_size)

    channel=1
    axarr[ind, channel ].imshow(feature_image[:, :, channel], cmap='gray')
    title = "ch {0}".format(channel)
    axarr[ind, channel].set_title(title, fontsize=font_size)
    axarr[ind, channel ].set_xticks([])
    axarr[ind, channel ].set_yticks([])

for indn, j in enumerate([i1, i2]):
    ind = indn + 2
    image = plt.imread(notcars_train[j])
    feature_image = cv2.cvtColor(image, colorspace)

    axarr[ind, 0].imshow(image)
    axarr[ind, 0].set_xticks([])
    axarr[ind, 0].set_yticks([])
    title = "not car {0}".format(j)
    axarr[ind, 0].set_title(title, fontsize=font_size)

    channel = 1
    axarr[ind, channel].imshow(feature_image[:, :, channel], cmap='gray')
    title = "ch {0}".format(channel)
    axarr[ind, channel].set_title(title, fontsize=font_size)
    axarr[ind, channel].set_xticks([])
    axarr[ind, channel].set_yticks([])

plt.show()
plt.savefig('./images/HOG_features_HLS.png')


pickle_file = 'ProcessedData.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')