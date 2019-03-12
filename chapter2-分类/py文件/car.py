# -*- coding:utf-8 -*- 
# Author: Roc-J

import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

if __name__ == '__main__':
    filename = sys.argv[1]
    X = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            data = line[:-1].split(',')
            X.append(data)

    X = np.array(X)

    # auto label
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        label_encoder.append(LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    # create RandomForestClassifier
    params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
    classifier = RandomForestClassifier(**params)

    classifier.fit(X, y)

    # cross validation
    accuracy = cross_validation.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
    print u'---------交叉验证-----------='
    print "Accuracy of the classifier: ", round(100.0 * accuracy.mean(), 2), '%'

    # predict a example
    input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
    input_data_encoded = [-1] * len(input_data)
    for i, item in enumerate(input_data):
        input_data_encoded[i] = label_encoder[i].transform(list([input_data[i]]))[0]

    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

    output_pred = classifier.predict(input_data_encoded)
    print "output class :", label_encoder[-1].inverse_transform(output_pred)[0]

    # 验证曲线
    classifier = RandomForestClassifier(max_depth=4, random_state=7)
    parameter_grid = np.linspace(25, 200, 8).astype(int)
    train_scores, validation_scores = validation_curve(classifier, X, y, "n_estimators", parameter_grid, cv=5)
    print "validation curve:"
    print "Param: n_estimations\n Training score:", train_scores
    print "Param: n_estimations\n Validation score: ", validation_scores

    # plot
    plt.figure()
    plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
    plt.title('Training curve')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.show()

    #
    print u'----------对max_depth验证----------'
    classifier = RandomForestClassifier(n_estimators=20, random_state=7)
    parameter_grid = np.linspace(2, 10, 5).astype(int)
    train_scores, valid_scores = validation_curve(classifier, X, y, "max_depth", parameter_grid, cv=5)
    print "\nParam: max_depth\n Training scores: ", train_scores
    print "\nParam: max_depth\n Validation scores: ", validation_scores

    plt.figure()
    plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color="black")
    plt.title('Validation curve')
    plt.xlabel('Maximun depth of the tree')
    plt.ylabel('Accuracy')
    plt.show()

    print u'--------生成学习曲线----------'
    classifier = RandomForestClassifier(random_state=7)

    parameter_grid = np.array([200, 500, 800, 1100])
    train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid, cv=5)
    print u'---------learning cure----------'
    print "\n Training scores: ", train_sizes
    print "\n Validation scores :", validation_scores

    plt.figure()
    plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
    plt.title('Learning curve')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.show()

