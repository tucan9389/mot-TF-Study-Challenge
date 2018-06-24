# -*- coding: utf-8 -*-
# ! /usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 1]
b_real = -0.2

noise = np.random.rand(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise


plt.plot(x_data, y_data[0], 'ro', alpha=0.1)
plt.ylabel('y_data')
plt.show()