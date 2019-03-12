# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # data X
    # classifier y
    X = np.array([
        [3, 1], [2, 5], [1, 8], [6, 4], [5, 2], [3, 5], [4, 7], [4, -1]
    ])

    y = [0, 1, 1, 0, 0, 1, 1, 0]

    # class the data
    class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    # plot
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], color='green', marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', marker='x')

    # draw the separator line
    x_line = range(10)
    y_line = x_line
    plt.plot(x_line, y_line, color='black', linewidth=3)
    plt.title('simple_classifier')
    plt.show()
