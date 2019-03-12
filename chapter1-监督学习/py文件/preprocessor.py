# -*- coding:utf-8 -*-
# Author: Roc-J

import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, -5.4],
    [0, 4, -0.3, 2.1],
    [1, 3.3, -1.9, -4.3]
])

# 均值移除
data_standardized = preprocessing.scale(data)
print "\nMean =", data_standardized.mean(axis=0)
print "Std deviation = ", data_standardized.std(axis=0)

# 范围缩放
'''
数据点中的每个特征的数值范围变化可能很大，因此有时将特征的数值范围缩放到合理的大小是非常重要的
'''
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print "\nMin max scaled data =", data_scaled

# 归一化
'''
数据归一化
机器学习中最常用的归一化形式就是将特征向量调整L1范数，使特征向量的数值之和为1

'''
data_normalized = preprocessing.normalize(data, norm='l1')
print "\nL1 normalized data=", data_normalized

# 二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print "\nBinarized data=", data_binarized