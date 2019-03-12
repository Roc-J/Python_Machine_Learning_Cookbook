# -*- coding:utf-8 -*- 
# Author: Roc-J

import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from logistic_regression import plot_classifier
from sklearn import cross_validation

def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [ float(i) for i in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])

    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__ == '__main__':
    print u'---------对所有数据进行建模-------------'
    filename = sys.argv[1]

    X, y = load_data(filename)

    # create GaussianNB
    classifier_gaussiannb = GaussianNB()
    classifier_gaussiannb.fit(X, y)
    y_pred = classifier_gaussiannb.predict(X)

    print u'--------模型准确率---------'
    accuracy = 100.0 * (y_pred == y).sum() / X.shape[0]
    print "Accuracy of the classifier: ", round(accuracy, 2), '%'

    # plot the data and edge
    plot_classifier(classifier_gaussiannb, X, y)

    print u'--------分为训练集和测试集---------'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

    classifier_gaussiannb_new = GaussianNB()
    classifier_gaussiannb_new.fit(X_train, y_train)

    # predict
    y_test_pred = classifier_gaussiannb_new.predict(X_test)

    accuracy = 100.0 * (y_test_pred == y_test).sum() / X_test.shape[0]
    print "Accuracy of the classifier is :", round(accuracy, 2), '%'

    plot_classifier(classifier_gaussiannb_new, X_test, y_test)

    # cross validation and scoring function
    num_validation = 5
    accuracy = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validation)
    print "Accuracy of the classifier: ", round(100.0 * accuracy.mean(), 2), '%'

    f1 = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validation)
    print "F1 score: ", round(100.0 * f1.mean(), 2), '%'

    precision = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validation)
    print "Precision score: ", round(100.0 * precision.mean(), 2), '%'

    recall = cross_validation.cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validation)
    print "Recall score: ", round(100.0 * recall.mean(), 2), '%'