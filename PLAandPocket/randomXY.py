import random

import numpy as np



def generateRandomXY(num):
    data = []
    labels = []
    for i in range(0, 20):
        X1 = random.randint(0,10)
        Y1 = random.randint(0,10)
        if X1 + Y1 - 10 > 0:
            label = 1
        else:
            label = -1
        data.append([X1, Y1])
        labels.append(label)
    return np.array(data), np.array(labels)


if __name__ == '__main__':
    data , label = generateRandomXY(10)
    print(label)