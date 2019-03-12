# -*- coding:utf-8 -*- 
# Author: Roc-J

import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

filename = sys.argv[1]

X = []
Y = []
print u'--------读取文件---------'
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)

print u'--------训练数据---------'
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

# 测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:]).reshape((num_test, 1))

# 创建线性回归对象
linear_regressor = linear_model.LinearRegression()

# 用训练数据集训练模型
linear_regressor.fit(X_train, Y_train)

print u'--------绘制图像---------'
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')

plt.show()

y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, y_test_pred)
plt.title('Test data')
plt.show()

print u'----------计算回归误差------------'
import sklearn.metrics as sm

print "Mean absolute error:", round(sm.mean_absolute_error(Y_test, y_test_pred), 2)
print "Mean squared error:", round(sm.mean_squared_error(Y_test, y_test_pred), 2)
print "Median absolute error:", round(sm.median_absolute_error(Y_test, y_test_pred), 2)
print "explained variance score:", round(sm.explained_variance_score(Y_test, y_test_pred), 2)
print "R2 score:", round(sm.r2_score(Y_test, y_test_pred), 2)

print u'---------保存模型数据------------'
import cPickle as pickle

output_model_file = 'saved_model.pkl'
with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f:
    model_linregre = pickle.load(f)

y_test_pred_new = model_linregre.predict(X_test)
print '\nNew mean absolute error =', round(sm.mean_absolute_error(Y_test, y_test_pred_new), 2)