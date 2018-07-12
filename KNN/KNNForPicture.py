import os
import operator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tackleDir(dir):
    """
    将整个文件夹中的文件读取出来，并返回数据集和对应的标签
    :param dir:
    :return:
    """
    labels = []
    vecDataList =[]
    for file in os.listdir(dir):
        filePath = os.path.join(dir, file)
        vecData, label = tackleFile(filePath)
        labels.append(label)
        vecDataList.append(vecData)

    dataSet = np.array(vecDataList)

    return dataSet, labels


def tackleFile(filePath):
    """
    将单个文件的数据转换成一维数组，并保存他的标签
    :param filePath:
    :return:
    """
    filePathPre = os.path.splitext(filePath)[0].split('_')[0]
    label = int(filePathPre.split('/')[-1])
    string = ""
    with open(filePath, "r") as f:
        for line in f.readlines():
            string += line.strip("\n")
    vecDataList = string.split(" ")
    vecData = [int(item) for item in vecDataList[0]]

    return vecData, label


def classify(dataSet, labels, k, testData):
    """
    对单个数据集进行分类
    :param dataSet:
    :param labels:
    :param k:
    :param testData:
    :return:
    """
    m = np.shape(dataSet)[0]
    testData = np.tile(testData, (m, 1))
    diffData = dataSet - testData
    squaData = diffData ** 2
    sumData = np.sum(squaData, axis=1)
    distances = sumData ** 0.5
    sortDistIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistIndex[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def errorRate(trainSet, trainLabels, testSet, testLabels, k):
    errors = 0
    testRows = np.shape(testSet)[0]
    for i in range(testRows):
        classifyVal = classify(trainSet, trainLabels, k, testSet[i, :])
        if classifyVal != testLabels[i]:
            errors += 1
    return errors / float(testRows)


def validateTest():
    trainSet, trainLabels = tackleDir('./trainingDigits')
    testSet, testLabels = tackleDir('./testDigits')
    errorList = []
    kList = []
    k = 3
    # while True:
    #     error = errorRate(trainSet, trainLabels, testSet, testLabels, k)
    #     errorList.append(error)
    #     kList.append(k)
    #     if min(errorList) < 0.01 or k > 3:
    #         return min(errorList), kList[errorList.index(min(errorList))]
    #     k += 1
    error = errorRate(trainSet, trainLabels, testSet, testLabels, k)
    return error


def start(testData, k):
    trainSet, trainLabels = tackleDir('./trainingDigits')
    classifyVal = classify(trainSet, trainLabels, k, testData)
    return classifyVal


def plotError(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("errorRate")
    plt.show()


def imageToMatrix(filename):
    im = Image.open(filename)
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float') / 255.0
    new_data = np.reshape(data, (32, 32))
    return new_data


if __name__ == "__main__":
    # vecData, label = tackleFile('./testDigits/7_44.txt')
    # testData = np.array(vecData)
    # k = 3
    # claVal = start(testData, k)
    # print(claVal, label)
    error = validateTest()
    print(error)