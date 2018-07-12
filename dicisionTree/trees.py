from math import log
import operator
import numpy as np
import matplotlib.pyplot as plt


def calcShanninEnt(dataSet):
    """
    计算信息熵的函数
    :param dataSet: 整个数据集
    :return:
    """
    dataNum = len(dataSet)
    labels = [data[-1] for data in dataSet]
    labelCounts = {}
    for label in labels:
        labelCounts[label] = labelCounts.get(label, 0) + 1
    shanNonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / dataNum
        shanNonEnt -= prob * log(prob, 2)
    return shanNonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    return dataSet


def splitDataSet(dataSet, featureIndex: int, value):
    """
    返回该属性的一个取值对应的数据集，该数据集不再包括该特征，用于下一步递归
    :param dataSet: 数据集
    :param featureIndex: 当前划分的属性
    :param value: 属性的某一个取值
    :return:
    """
    subDataSet = []
    for data in dataSet:
        if data[featureIndex] == value:
            reducedFeatureData = data[:featureIndex] + data[featureIndex + 1:]
            subDataSet.append(reducedFeatureData)
    return subDataSet


def chooseBestFeatureToSplit(dataSet) -> int:
    """
    计算各属性的信息增益，得到最佳的属性
    :param dataSet: 数据集
    :return:
    """
    featureIndex = len(dataSet[0]) - 1  # 获取特征的索引，以索引值来代表每个特征
    baseShanNonEnt = calcShanninEnt(dataSet)  # 计算整个数据集的信息熵
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for index in range(featureIndex):
        featList = [data[index] for data in dataSet]  # 获取该特征所有的值
        uniqueVals = set(featList)
        featureEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, index, value)
            prob = len(subDataSet) / len(dataSet)
            featureEnt += prob * calcShanninEnt(subDataSet)
        infoGain = baseShanNonEnt - featureEnt
        if infoGain > bestInfoGain:  # 获取最好的信息增益
            bestInfoGain = infoGain
            bestFeatureIndex = index

    return bestFeatureIndex


def voteMajority(classList):
    """
    当所有属性都被处理完后，其叶子节点的数据仍不属于同一类，此时需要通过投票来决定该叶子节点的类别
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    根据数据集和标签创建决策树，决策树的叶子节点对应的是分类的类别，其余的节点都是属性值
    :param dataSet:
    :param labels:对应的属性列表，可以看作是列索引对应的值
    :return:
    """
    if not dataSet:
        return
    classList = [data[-1] for data in dataSet]
    if len(set(classList)) == 1:
        return classList[0]
    if len(dataSet[0]) == 1:
        return voteMajority(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeature])  # 删除已经执行过得属性
    featureValues = [data[bestFeature] for data in dataSet]
    uniqueValues = set(featureValues)  # 得到当前属性的取值
    for value in uniqueValues:
        # 当前属性的每个取值对应着一个子数据集，对该数据集执行上述逻辑，依次递归
        subLabels = labels[:]  # 赋值一个labels的列表作为参数进行传递
        # 调用splitDataSet函数来对当前的数据集进行分割，对分割后的数据集递归调用createTree函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)

    return myTree


def classify(inputTree, featLabel, testVec):
    """
    对待预测的数据进行分类
    :param inputTree: 建立好的决策树
    :param featLabel: 属性的索引列表
    :param testVec: 待测试的数据
    :return:
    """
    classLabel= -1
    firstStr = [key for key in inputTree.keys()][0]
    secondDict = inputTree[firstStr]
    print(featLabel)
    print(firstStr)
    featIndex = featLabel.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabel, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


myTree = grabTree('./lensesTree.txt')
print(myTree)

