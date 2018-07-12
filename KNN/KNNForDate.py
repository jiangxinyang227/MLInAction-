import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split


filename = './data/datingTestSet2.txt'


def fileToMatrix():
    dataSet = np.loadtxt(filename, usecols=(0, 1, 2))
    labels = np.loadtxt(filename, dtype=int, usecols=(3,))

    return dataSet, labels


def fileToMatrix1(ratio: float):
    """
    切割数据为验证集和训练集
    :param ratio: 用于验证的数据集比例
    :return:
    """
    dataSet = np.loadtxt(filename, usecols=(0, 1, 2))
    labels = np.loadtxt(filename, dtype=int, usecols=(3,))
    trainDataSet, testDataSet, trainLabel, testLabel = train_test_split(dataSet, labels, test_size=ratio)
    return trainDataSet, testDataSet, trainLabel, testLabel


def autoNorm(dataSet):
    """
    归一化处理
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet


def classify(dataset, labels, k: int, testData) -> str:
    """
    分类函数
    :param dataset: 训练集样本
    :param labels: 训练集标签
    :param k: KNN中的k值
    :param testData: 要分类的目标样本
    :return: 目标样本的类别
    """
    rows = np.shape(dataset)[0]
    diffDataSet = np.tile(testData, (rows, 1)) - dataset
    sqDiffDataSet = diffDataSet ** 2
    sqDistances = sqDiffDataSet.sum(axis=1)
    distances = sqDistances ** 0.5
    sortDistIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistIndex[i]]
        classCount[label] =classCount.get(label, 0) + 1
    # sortedClassCount: [(), (), ...]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def getError(dataSetNorm, labels, k, ratio=0.1):
    """
    未采用sklearn中的方法进行随机分割训练集和验证集
    :param dataSetNorm: 整体样本
    :param labels: 样本标签
    :param k: k值
    :param ratio: 分割验证集的比例
    :return: 验证集的错误率
    """
    errors = 0
    m = np.shape(dataSetNorm)[0]
    testNum = int(ratio * m)
    for i in range(testNum):
        testData = dataSetNorm[i, :]
        trainData = dataSetNorm[testNum:m, :]
        trainLabels = labels[testNum:m]
        classifyVal = classify(trainData, trainLabels, k, testData)
        if classifyVal != labels[i]:
            errors += 1
    return errors / float(testNum)


def getError1(trainNorm, trainLabel, testNorm, testLabel, k):
    """
    预先分割好训练集和测试集
    :param trainNorm: 训练集
    :param trainLabel: 训练集标签
    :param testNorm: 测试集
    :param testLabel: 测试集标签
    :param k: k值
    :return: 测试纸的分类错误率
    """
    errors = 0
    testRows = np.shape(testNorm)[0]
    for i in range(testRows):
        classifyVal = classify(trainNorm, trainLabel, k, testNorm[i, :])
        if classifyVal != testLabel[i]:
            errors += 1
    return errors / float(testRows)


def plotErroe(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.xlabel('k')
    plt.ylabel('errorRate')
    plt.show()


def test():
    trainDataSet, testDataSet, trainLabel, testLabel = fileToMatrix1(0.1)
    trainNorm = autoNorm(trainDataSet)
    testNorm = autoNorm(testDataSet)
    k = 8
    errorRate = getError1(trainNorm, trainLabel, testNorm, testLabel, k)
    return errorRate


def test1():
    """
    循环测试不同的k的取值下，样本分类的错误率
    :return:
    """
    trainDataSet, testDataSet, trainLabel, testLabel = fileToMatrix1(0.1)
    trainNorm = autoNorm(trainDataSet)
    testNorm = autoNorm(testDataSet)
    k = 3
    errorRateList = []
    kList = []
    while True:
        errorRate = getError1(trainNorm, trainLabel, testNorm, testLabel, k)
        if errorRate < 0.01 or k > 20:
            return min(errorRateList), kList[errorRateList.index(min(errorRateList))]
        errorRateList.append(errorRate)
        kList.append(k)
        k += 1


def getK():
    ks = []
    for i in range(100):
        errorRate, k = test1()
        ks.append(k)
    k = int(sum(ks) / len(ks))
    return k


def main(testData):
    dataSet, labels = fileToMatrix()
    k = 8
    classfilyVal = classify(dataSet, labels, k, testData)
    return classfilyVal





