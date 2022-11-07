import numpy as np


def pocket(data, label, epoch):
    data_num, feature_num = data.shape
    w = np.zeros(feature_num + 1)
    a = np.ones([data_num, 1])
    data = np.concatenate((a, data), 1)
    for j in range(epoch):
        for i in range(data_num):
            if(1):
                y = np.sign(w@data[i])
                if y != label[i]:
                    w_ = w + label[i] * data[i]
                    if predict(data, label, w) > predict(data, label, w_):
                        w = w_
    return w


def predict(data, label, w):
    num = data.shape[0]
    error = 0
    for i in range(num):
        if label[i]*w@data[i] <= 0:
            error += 1
    return error
