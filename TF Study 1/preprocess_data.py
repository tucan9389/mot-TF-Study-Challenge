#-*- coding: utf-8 -*-
#! /usr/bin/env python
# ------------
# 한글 주석을 달려면 위 전처리기 필요

# 사용 모델에 필요한 인풋 전처리

import cv2

# 28*28*1 array로 만들기
def get_image_data(path, data_width, data_height, data_channel):
    image = cv2.imread(path, 0)
    image = image.reshape((data_width, data_height, data_channel))
    return image  # numpy.ndarray


# categorical 데이터는 OneHot Encoding 필요