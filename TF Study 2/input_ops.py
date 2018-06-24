# 사용 모델 인풋 형식에 맞게 변환 (여러 데이터를 쉽게 바꿔끼고 섞을 수 있어야한다.)

from sklearn.model_selection import train_test_split
import numpy as np

def my_train_test_split(dataset):
    # X: 이미지 경로 배열 (예. '0/4123.png')
    # y: 이미지 레이블 배열 (예. '0')
    X = dataset[:, 0]
    y = np.array([int(y) for y in dataset[:, 1]])

    # train과 test로 데이터로 나누기
    # random_state를 지정하지 않으면 매번 다르게 train과 test를 나눠줌
    # train, test로 나누기
    return train_test_split(X, y, test_size = 0.2)#, random_state=0)

# print(X_train[0:3])
# print(y_train[0:3])