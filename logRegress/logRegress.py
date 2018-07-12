from math import exp, log
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


filename = './data/testSet.txt'


def loadDataSet(filename):
    """
    将文本数据解析成numpy中的数组类型, 返回的是array数据类型
    :return:
    """
    dataSet = np.loadtxt(filename)
    labels = np.loadtxt(filename, usecols=(-1,))
    oneData = np.ones(len(dataSet))
    dataSet = np.c_[oneData, dataSet]  # 对所有的数据都加上x0=1
    dataSet = np.delete(dataSet, -1, axis=1)
    return dataSet, labels


def sigmoid(value):
    """
    sigmoid函数
    :param value:
    :return:
    """
    value = np.array(value)
    return 1 / (1 + np.exp(-value))


def getGrandient(dataMatrix, labelMat, w, labda):
    """
    计算当前的梯度, 直接进行矩阵的操作，得到所有的值所计算的向量的平均值
    :param dataMatrix:
    :param labelMat:
    :param w:
    :return:
    """
    rows = np.shape(dataMatrix)[0]
    z = np.array(dataMatrix * w)  # 转换成array的数组类型，便于对里面的单个元素进行同样的操作
    innerValue = (1 / (1 + np.exp(-z)) - np.array(labelMat))
    deltaW = (dataMatrix.transpose() * np.mat(innerValue) + labda * w) / rows
    return deltaW


def getRandGrand(dataMatrix, labelMat, w, labda, n):
    """
    随机梯度下降，从样本中选择若干个数据来计算当前的梯度
    :param dataMatrix: 数据集
    :param labelMat: 数据集的标签
    :param w: 当前的权重向量值
    :param labda: 正则化项的参数
    :param n:  要取出用于计算的子集的个数
    :return:
    """
    restData, calcuData, restLabels, calcuLabels = train_test_split(np.array(dataMatrix), np.array(labelMat),
                                                                    test_size=n/np.shape(dataMatrix)[0])
    calcuMatrix = np.mat(calcuData)
    rows = np.shape(calcuMatrix)[0]
    z = np.array(calcuMatrix * w)  # 转换成array的数组类型，便于对里面的单个元素进行同样的操作
    innerValue = (1 / (1 + np.exp(-z)) - calcuLabels)  # 数组间映射的计算
    deltaW = (calcuMatrix.transpose() * np.mat(innerValue) + labda * w) / rows
    return deltaW


def granDescent(dataSet, labels, maxCycles):
    """
    采用梯度下降法求最佳的参数权重w
    :param dataSet:
    :param labels:
    :return:
    """
    w = np.mat(np.ones(np.shape(dataSet)[1])).transpose()  # 随机生成值处于0-1之间的初始权重向量, 并转换成矩阵，便于后面计算
    labda = 0.1  # 正则化项参数
    # maxCycles = 200  # 最大的迭代数
    dataMatrix = np.mat(dataSet)  # 将数据转换成矩阵的格式
    labelMat = np.mat(labels).transpose()  # 将行向量转化成列向量
    wArray = [np.array(w.transpose()).tolist()[0]]
    for i in range(maxCycles):
        alpha = 3 / (i + 1) + 0.01  # 设置学习速率α的值
        deltaW = getGrandient(dataMatrix, labelMat, w, labda)
        w = w - alpha * deltaW
        array = np.array(w.transpose()).tolist()[0]
        wArray.append(array)
    return w


def classify(dataSet, labels, maxCycles, testVect):
    """
    用于对数据进行分类
    :param dataSet:
    :param labels:
    :param maxCycles:
    :param testVect: numpy数组类型
    :return:
    """
    testVect = np.mat(testVect)
    w = granDescent(dataSet, labels, maxCycles)
    value = -(np.array(testVect * w).tolist()[0][0])
    prob = 1 / (1 + exp(round(value, 2)))
    if prob >= 0.5:
        return 1
    else:
        return 0


def testError(dataSet, labels, maxCycles, testData, testLabels):
    """

    :param dataSet:
    :param labels:
    :param maxCycles:
    :param testData:
    :return:
    """
    rows = np.shape(testData)[0]
    errors = 0
    for i in range(rows):
        classifyValue = classify(dataSet, labels, maxCycles, testData[i, :])
        if classifyValue != testLabels[i]:
            errors += 1

    return errors / rows


def plotBestFit(dataSet, labels, maxCycles):
    w = granDescent(dataSet, labels, maxCycles)
    w = np.array(w)
    rows = np.shape(dataSet)[0]
    xcord0 = []; ycord0 = []  # 在这里x，y分别对应着二维数据中的两个特征的值
    xcord1 = []; ycord1 = []  # 0，1是用于类别的分类，在作图数用来实现点的形状不一致
    for i in range(rows):
        if labels[i] == 1:
            xcord1.append(dataSet[i, 1])
            ycord1.append(dataSet[i, 2])
        else:
            xcord0.append(dataSet[i, 1])
            ycord0.append(dataSet[i, 2])
    fig = plt.figure()

    # ax1 = fig.add_subplot(2, 2, 1)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    ax = fig.add_subplot(111)

    ax.scatter(xcord0, ycord0, s=30, c="red", marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-w[0] - w[1] * x) / w[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')

    # w0 = wArray[:, 0].tolist()
    # w1 = wArray[:, 1].tolist()
    # w2 = wArray[:, 2].tolist()
    # epochList = [epoch for epoch in range(epoch + 1)]
    # ax1.plot(epochList, w0, label='w0', color='red')
    # ax2.plot(epochList, w1, label='w1', color='green')
    # ax3.plot(epochList, w2, label='w2', color='blue')
    plt.show()


def dataNorm(dataSet):
    """
    归一化处理
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    minVals[0] = 0
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet


def train():
    testData, testLabels = loadDataSet('./data/horseColicTest.txt')
    trainData, trainLabels = loadDataSet('./data/horseColicTraining.txt')
    trainNorm = dataNorm(trainData)
    testNorm = dataNorm(testData)
    maxCycles = 1000
    # error = 1
    # for i in range(maxCycles):
    #     error = testError(trainNorm, trainLabels, maxCycles, testNorm, testLabels)
    #     if error < 0.1:
    #         return error, maxCycles
    error = testError(trainNorm, trainLabels, maxCycles, testNorm, testLabels)
    return error


print(train())

