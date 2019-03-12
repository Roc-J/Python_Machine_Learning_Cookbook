# -*- coding:utf-8 -*- 
# Author: Roc-J

import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures


filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        y.append(yt)

# traing/test split
num_training = int(0.8 * len(X))
num_testing = len(X) - num_training

# training data
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

# test data
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

# create the regression model
linear_regression = linear_model.LinearRegression()
ridge_regression = linear_model.Ridge(alpha=0.3, fit_intercept=True, max_iter=10000)

# traing the model
linear_regression.fit(X_train, y_train)
ridge_regression.fit(X_train, y_train)

# predict the value
y_test_pred = linear_regression.predict(X_test)
y_test_pred_ridge = ridge_regression.predict(X_test)

# measure the model
print "Linear model measure:"
print "Mean absolute error:", round(sm.mean_absolute_error(y_test, y_test_pred), 2)
print "Mean squared error:", round(sm.mean_squared_error(y_test, y_test_pred), 2)
print "Median absolute error:", round(sm.median_absolute_error(y_test, y_test_pred), 2)
print "Explained variance error: ", round(sm.explained_variance_score(y_test, y_test_pred), 2)
print "R2 score: ", round(sm.r2_score(y_test, y_test_pred), 2)

print "Ridge model measure:"
print "Mean absolute error:", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2)
print "Mean squared error:", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2)
print "Median absolute error:", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2)
print "Explained variance error: ", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2)
print "R2 score: ", round(sm.r2_score(y_test, y_test_pred_ridge), 2)


# Polynomail regression
polynomial = PolynomialFeatures(degree=15)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = np.array([0.39, 2.78, 7.11]).reshape(1, -1)
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print "\nLinear regression:\n", linear_regression.predict(datapoint)
print "\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint)