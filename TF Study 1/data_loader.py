#-*- coding: utf-8 -*-
#! /usr/bin/env python
# ------------
# 한글 주석을 달려면 위 전처리기 필요

# 데이터를 빠르게 불러와 메모리에 로드

import os
import numpy as np

# 경로로부터 path와 label 조합의 numpy.ndarray
def my_read_dataset_path_and_label(path = './data/mnist/'):
    labels = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

    for label in labels:
        image_filenames = os.listdir(os.path.join(path, label))
        dataset.extend([ [label + "/" + image_filename, label] for image_filename in image_filenames ])
    return np.array(dataset)