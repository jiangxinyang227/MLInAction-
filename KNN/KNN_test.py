import numpy as np
import operator
import math
from collections import Counter


def createDateSet():
    X = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
    ])

    Y = ['A', 'A', 'B', 'B']

    return X, Y


def classify(dataset, label, k, testData):
    distances = []
    rows = len(dataset[:, 0])
    for i in range(rows):
        rowData = dataset[i, :]
        sum = 0
        for j in range(len(rowData)):
            sum += (testData[j] - rowData[j]) ** 2

        distance = math.sqrt(sum)
        distances.append((distance, i))

    sortDistances = sorted(distances, key=operator.itemgetter(0), reverse=True)

    kSortDistances = sortDistances[:k]
    kLabels = [label[item[1]] for item in kSortDistances]
    return countMax(kLabels)


def countMax(kLabels):
    count = {}
    for item in kLabels:
        if item in count:
            count[item] += 1
        count[item] = 1
    classify= max(count.items(), key=lambda x: x[1])[0]
    return classify