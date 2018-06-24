#-*- coding: utf-8 -*-
#! /usr/bin/env python

from data_loader import my_read_dataset_path_and_label
from input_ops import my_train_test_split
from preprocess_data import get_image_data
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt

class DataManager(object):
    def __init__(self, dataset_path, batch_size=32, data_width=28, data_height=28, data_channel=1):
        self.dataset_path = dataset_path

        self.batch_size = batch_size

        self.data_width = data_width
        self.data_height = data_height
        self.data_channel = data_channel

        self.labelencoder = LabelEncoder()
        self.onehotencoder = OneHotEncoder(categorical_features=[0])

        self.batch_index = 0

        self.load_path_labels_from_dataset()

    def load_path_labels_from_dataset(self):
        # dataset 읽어오기
        dataset = my_read_dataset_path_and_label(self.dataset_path)
        # print("-------------------------------------")
        # print(dataset)

        # train, test로 나누기
        self.X_train, self.X_test, self.y_train, self.y_test = my_train_test_split(dataset)

        # 전체 batch 갯수 설정
        self.total_batch = len(self.X_train)/self.batch_size

        # OneHotEncoder
        self.onehotencoder.fit(self.y_train.reshape(-1, 1))


    def next_batch(self):
        batch_index = self.batch_index
        self.batch_index += 1
        if self.batch_index == self.total_batch:
            self.batch_index = 0
        return self.get_batch_data(batch_index=batch_index)

    def get_batch_data(self, batch_index=0):
        batch_index = batch_index % int(self.total_batch)
        start_index = batch_index*self.batch_size
        X = self.X_train[start_index:start_index + self.batch_size]
        y = self.y_train[start_index:start_index + self.batch_size]

        return self.get_input_data(X, y)

    def get_input_data(self, X, y):
        if (len(X) != len(y)):
            return None
        batch_size = len(X)
        one_batch_X = np.zeros(shape=(batch_size, self.data_width, self.data_height, self.data_channel))
        for i in range(self.batch_size):
            one_batch_X[i] = get_image_data(self.dataset_path + X[i], self.data_width, self.data_height, self.data_channel)
        one_batch_y = self.onehotencoder.transform(y.reshape(-1, 1)).toarray()

        return one_batch_X, one_batch_y

    def get_test_batch_data(self, test_batch_size=50):
        start_index = 0
        X = self.X_test[start_index:start_index + test_batch_size]
        y = self.y_test[start_index:start_index + test_batch_size]

        return self.get_input_data(X, y)



# batch_idx = 0
# idx = 2
#
#
# data_manager = DataManager('data/mnist/')
# data_manager.prepare_data()
# X, y = data_manager.get_input_data(batch_idx)
#
# print("len(X): {}".format(len(X)))
# print("len(y): {}".format(len(y)))
#
#
# my_X = X[idx]
# my_y = y[idx]
#
# # The first column is the label
# label = str(np.where(my_y==1)[0][0])
#
# # The rest of columns are pixels
# pixels = my_X#.reshape(28,28)
#
# # Make those columns into a array of 8-bits pixels
# # This array will be of 1D with length 784
# # The pixel intensity values are integers from 0 to 255
# pixels = np.array(pixels, dtype='uint8')
#
# # Reshape the array into 28 x 28 array (2-dimensional array)
# pixels = pixels.reshape((28, 28))
#
# # Plot
# plt.title('Label is {label}'.format(label=label))
# plt.imshow(pixels, cmap='gray')
# plt.show()