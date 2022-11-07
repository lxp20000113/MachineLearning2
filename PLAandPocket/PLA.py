import numpy as np


def PLA(data, label):
    data_num, feature_num = data.shape
    w = np.zeros(feature_num + 1)
    a = np.ones([data_num, 1])
    data = np.concatenate((a, data), 1)
    terminated = False
    while not terminated:
        false_num = 0
        for i in range(data_num):
            if not terminated:
                y = np.sign(w@data[i])
                if y != label[i]:
                    w += label[i]*data[i]
                    false_num += 1
        if false_num == 0:
            terminated = True
    return w

