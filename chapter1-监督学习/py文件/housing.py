# -*- coding:utf-8 -*- 
# Author: Roc-J

from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import sklearn.metrics as sm
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(feature_importances, title, feature_names):
    # 将重要性标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # 将得分从高到底排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # load dataset
    housing_data = load_boston()

    # shuffle
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # train/test count
    num_training = int(0.8*len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # create decisiontree
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # create decision tree regression model with AdaBoost
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    print u'------输出评估结果----------'
    print 'Decisiontree regression'
    y_pred_dt = dt_regressor.predict(X_test)
    print "Mean_squared_error: ", round(sm.mean_squared_error(y_test, y_pred_dt), 2)
    print "Explained_variance_score: ", round(sm.explained_variance_score(y_test, y_pred_dt), 2)

    print 'Decisiontree regression with Adaboost'
    y_pred_ab = ab_regressor.predict(X_test)
    print "Mean_square_error: ", round(sm.mean_squared_error(y_test, y_pred_ab), 2)
    print "Explained_variance_score: ", round(sm.explained_variance_score(y_test, y_pred_ab), 2)

    # compute the importance of features
    plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names)
