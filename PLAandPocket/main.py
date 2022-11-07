# This is a sample Python script.
from copy import copy

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from PLA import *
from Pocket import *
from randomXY import *


#  网上找一个显示函数
def plot_dots(data, label, w , id):
    data_num, feature_num = data.shape
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(data_num):
        if (label[i] == 1):
            xcord1.append(data[i, 0])
            ycord1.append(data[i, 1])
        else:
            xcord2.append(data[i, 0])
            ycord2.append(data[i, 1])
    plt.figure(id)
    if id == 1:
        plt.title("PLA")
    else:
        plt.title("Pocket")
    plt.scatter(xcord1, ycord1, s=40, c='red', marker='s')
    plt.scatter(xcord2, ycord2, s=40, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 绘制分类线
    x = np.arange(3, 8.0, 0.1)
    y = (-w[0] - w[1] * x) / w[2]
    plt.plot(x, y)
    plt.pause(10)


def plot_lines(w):
    x = range(-3.0, 8.0, 0.1)
    y = (-w[0] - w[1] * x) / w[2]
    plt.plot(x, y)


def RunCompare(data, label, iter_num=200):
    poc_data , poc_label = copy(data), copy(label)
    w = PLA(data, label)
    plot_dots(data, label, w, 1)
    w = pocket(poc_data, poc_label, iter_num)

    plot_dots(poc_data, poc_label, w, 2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, label = generateRandomXY(10)
    RunCompare(data, label)

# See PyCharm help at https://www.jetbrains.com/help
