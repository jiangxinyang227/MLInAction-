import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
    """
    该方法加载速度更快，但唯一要注意的是读取数据时，对每个数据要用float函数去转变成矩阵可用的数值类型
    :param filename:
    :return:
    """
    fr = open(filename)
    numFeat = len(fr.readline().split('\t')) - 1  # 读取出当前数据的特征数
    dataMat = []
    labelsMat = []
    for line in fr.readlines():
        lineArr = [float(line.strip().split('\t')[i]) for i in range(numFeat)]
        dataMat.append(lineArr)
        labelsMat.append(float(line.strip().split('\t')[-1]))

    return np.array(dataMat), labelsMat


def createData(filename):
    """
    从文件中读取数据，最好是用这种方法读取数据，因为在读取的过程中会将数组中的元素都转换成numpy数值的形式，有利于后面的矩阵运算, 但比较耗时
    :param filename:
    :return:
    """
    dataSet = np.loadtxt(filename)
    labels = np.loadtxt(filename, usecols=(-1,))
    dataSet = np.delete(dataSet, -1, axis=1)
    return dataSet, labels


def standRegress(dataSet, labels):
    """
    利用正规方程式求解w的值，在求解的过程中需要判断xTx矩阵的可逆性
    :param dataSet:
    :param labels:
    :return:
    """
    dataMatrix = np.mat(dataSet)
    labelsMat = np.mat(labels).T
    xTx = dataMatrix.T * dataMatrix
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * dataMatrix.T * labelsMat
    return ws


def plotFig():
    dataSet, labels = loadData('./data/ex0.txt')
    ws = standRegress(dataSet, labels)
    dataSet = np.mat(dataSet)
    labels = np.mat(labels)
    preY = dataSet * np.mat(ws)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  dataSet[:, 1].flatten().A[0]可以将列矩阵平铺展开，然后通过A[0]来进行切片出列表
    ax.scatter(dataSet[:, 1].flatten().A[0], labels.T[:, 0].flatten().A[0], color='green')
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax.plot(dataSet[:, 1], preY[:, 0], color='red')
    plt.show()


def lwlr(testPoint, dataSet, labels, k=1.0):
    """
    局部线性加权，用于解决欠拟合的问题，每次回归时都要重新计算权重向量，算法太慢了，不合适
    :param testPoint:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataMatrix = np.mat(dataSet)
    labelsMat = np.mat(labels).T
    m = np.shape(dataMatrix)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - dataMatrix[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / -2.0 * k ** 2)
    xTx = dataMatrix.T * (weights * dataMatrix)
    if np.linalg.det(xTx) == 0.0:  # 如果矩阵不可逆，则直接返回
        return
    ws = xTx.I * dataMatrix.T * (weights * labelsMat)
    np.shape(ws)
    return testPoint * ws


def lwlrTest(testData, dataSet, labels, k=1.0):
    testMatrix = np.mat(testData)
    m = np.shape(testData)[0]
    testPre = np.zeros(m)
    for i in range(m):
        pre = lwlr(testMatrix[i, :], dataSet, labels)
        testPre[i] = pre
    return testPre


def errorRate(yActual, yPredict):
    yActual = np.array(yActual)
    yPredict = np.array(yPredict)
    return ((yActual - yPredict) ** 2).sum()


def test():
    dataSet, labels = loadData('./data/abalone.txt')
    yAvtual = np.mat(labels).T
    yPredict1 = lwlrTest(dataSet, dataSet, labels, k=0.001)

    # yPredict2 = np.mat(dataSet) * np.mat(standRegress(dataSet, labels))
    error1 = errorRate(yAvtual, yPredict1)
    # error2 = errorRate(yAvtual, yPredict2)
    print(error1)
    # print(error2)
test()



