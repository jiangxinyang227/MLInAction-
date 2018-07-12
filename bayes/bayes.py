from math import log
import os
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split


def createFeatList(dataSet):
    """
    将数据集中所有的单词以无重复的形式排列在列表中，作为属性集
    :param dataSet: 数据集
    :return: 属性集的列表
    """
    featureSet = set([])
    for data in dataSet:
        featureSet = featureSet | set(data)
    return list(featureSet)


def wordsToVec(featureSet, singleData):
    """
    将原始的数据向量化，属性在featureSet中则对应的索引位置加1，但也可以看出对于重复出现的单词也只会赋值1，
    属性不存在或者属性不在featureSet中就默认为0
    :param featureSet:
    :param singleData:
    :return:
    """
    dataVec = [0] * len(featureSet)
    for data in singleData:
        if data in featureSet:
            dataVec[featureSet.index(data)] = 1
    return dataVec


def wordsToVec1(featureSet, singleData):
    """
    将原始的数据向量化，对于重复出现的词会叠加在一个属性上
    :param featureSet:
    :param singleData:
    :return:
    """
    dataVec = [0] * len(featureSet)
    for data in singleData:
        if data in featureSet:
            dataVec[featureSet.index(data)] += 1
    return dataVec


def wordsToMatrix(dataSet):
    """
    将原始的数据集转换成稀疏矩阵
    :param dataSet: 原始数据集
    :return:
    """
    dataMatrix = []
    featureSet = createFeatList(dataSet)
    for data in dataSet:
        dataMatrix.append(wordsToVec1(featureSet, data))
    return dataMatrix


def trainNB0(trainData, trainLabels):
    """
    该函数只针对二分类问题， 获得正类别概率和各属性的条件概率, 对于之后的类别分类，直接从样本计算出的条件概率和类别概率中去拿所需要的值
    :param trainData:
    :param trainLabels:
    :return:
    """
    numTrainData = len(trainData)
    numWords = len(trainData[0])
    pAbusive = sum(trainLabels) / float(numTrainData)
    p0Num = np.ones(numWords)  # 初始化为1是为了避免出现0值，导致最后概率叠乘时出现概率为0
    p1Num = np.ones(numWords)
    p0Denom = 2.0  # 相应的对总属性数分母也加上一个值
    p1Denom = 2.0
    for i in range(numTrainData):
        if trainLabels[i] == 1:
            p1Num += trainData[i]  # 将每个样本的属性一一对应的加入到属性列表的各个索引值上
            p1Denom += sum(trainData[i])  # 将类别为1的样本中的属性个数相加在一起
        else:
            p0Num += trainData[i]
            p0Denom += sum(trainData[i])
    p0Vect = np.log(p0Num / p0Denom)  # 获得属性列表中各属性的条件概率, 取对数便于之后计算
    p1Vect = np.log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive


def classify(testData, p0Vect, p1Vect, pAbusive, featureList):
    """
    进行样本的分类， testData的数据类型必行是np.array
    :param testData:待分类的数据, 传入已经向量化了的数据
    :param p0Vect:类别为0所对应的各属性的条件概率的集合
    :param p1Vect:类别为1所对应的各属性的条件概率的集合
    :param pAbusive:正类别1的概率
    :return:
    """
    testVect = wordsToVec1(featureList, testData)
    testVect = np.array(testVect, dtype="float")

    p1 = np.sum(testVect * p1Vect) + log(pAbusive)  # 对于np.array数组相乘时，会对各索引对应的值相乘
    p0 = np.sum(testVect * p0Vect) + log(1-pAbusive)
    if p1 > p0:
        return 1
    return 0


def train(dataSet, labels):
    dataMatrix = wordsToMatrix(dataSet)
    featureList = createFeatList(dataSet)
    p0Vect, p1Vect, pAbusive = trainNB0(dataMatrix, labels)

    return p0Vect, p1Vect, pAbusive, featureList


def textParse(bigString: str) -> list:
    """
    将一个长字符串进行分割成由单个单词组成的列表, 将长度小于或等于2的字符串去掉
    :return:
    """
    regEx = re.compile(r'\W*')  # 按非字母、数字、下划线分割
    listOfTokens = regEx.split(bigString)
    listOfTokens = [token.lower() for token in listOfTokens if len(token) > 2]
    return listOfTokens


def getOriginalData(dir):
    originalDataSet = []
    for file in os.listdir(dir):
        bigString = ""
        with open(os.path.join(dir, file), 'rb') as f:
            for line in f.readlines():
                bigString = bigString + " " + str(line)
            listOfTokens = textParse(bigString)
            originalDataSet.append(listOfTokens)
    return originalDataSet


def getData():
    hamDataSet = getOriginalData('./ham')
    hamLabels = [1] * len(hamDataSet)
    spamDataSet = getOriginalData('./spam')
    spamLabels = [0] * len(spamDataSet)
    dataSet = hamDataSet + spamDataSet
    labels = hamLabels + spamLabels

    return dataSet, labels


def getTrainAndTest(ratio):
    dataSet, labels = getData()
    npDataSet = np.array(dataSet)
    npLabels = np.array(labels)
    trainData, testData, trainLabels, testLabels = train_test_split(npDataSet, npLabels, test_size=ratio)
    trainData = trainData.tolist()
    testData = testData.tolist()
    trainLabels = trainLabels.tolist()
    testLabels = testLabels.tolist()

    return trainData, testData, trainLabels, testLabels


def getError(testDataSet, testLabels, p0Vect, p1Vect, pAbusive, featureList):
    errors = 0
    for i in range(len(testDataSet)):
        classifyVal = classify(testDataSet[i], p0Vect, p1Vect, pAbusive, featureList)
        if classifyVal != testLabels[i]:
            errors += 1
    return errors / len(testDataSet)


def spamTest(ratio, n):
    errorList = []
    for i in range(n):
        trainData, testData, trainLabels, testLabels = getTrainAndTest(ratio)
        p0Vect, p1Vect, pAbusive, featureList = train(trainData, trainLabels)
        error = getError(testData, testLabels, p0Vect, p1Vect, pAbusive, featureList)
        errorList.append(error)
    return sum(errorList) / len(errorList)


meanError = spamTest(0.2, 100)
print(meanError)
