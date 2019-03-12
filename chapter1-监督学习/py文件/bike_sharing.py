# -*- coding:utf-8 -*- 
# Author: Roc-J
import csv
import sys

import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as sm
from housing import plot_feature_importances

def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:14])
        y.append(row[-1])

    # 提取特征名字
    feature_name = np.array(X[0])

    # 将第一行特征名称移除，仅保留数值
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_name

if __name__ == '__main__':
    # load data
    X, y, feature_names = load_dataset(sys.argv[1])
    X, y = shuffle(X, y, random_state=7)

    # train/test
    num_training = int(0.9*len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # create ramdomForestRegressor
    '''
    n_estimators 表示评估器的数量，表示随机森林需要使用的决策树的数量
    max_depth 是指每个决策树的最大深度
    min_samples_split是指决策树分裂一个节点需要用到的最小数据样本量
    '''
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # measure
    print u'-----------评估---------'
    y_pred = rf_regressor.predict(X_test)
    print "Mean squared error :", round(sm.mean_squared_error(y_test,y_pred), 2)
    print "Explained variance error: ", round(sm.explained_variance_score(y_test, y_pred), 2)

    # plot
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)