import numpy as np


def loadSimpData():
    dataMat = np.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过某特征的划分点（阀值）进行分类，可以分为-1和+1这两类
    :param dataMatrix: 原始数据集
    :param dimen: 特征属性
    :param threshVal: 划分点（阀值）
    :param threshIneq:
    :return:
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化属性集对应的类别数组
    if threshIneq == "lt":
        retArray[dataMatrix[:, dimen] < threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1

    return retArray


def buildStump(dataSet, labels, D):
    """
    因为是获得单层决策树，因此只要找到最佳属性和最佳划分点，使得分类器的错误率最低
    :param dataSet: 原始的数据集
    :param labels: 原始的数据集对应的分类
    :param D: 数据集上各样本的权重的集合
    :return:返回的值是最佳树桩的信息，最小的分类误差，分类后的分类标签，切返回的误差可以用来设置分类器的权重，更新样本的向量
    """
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10  # 设定的准备判断的划分点的个数
    bestStump = {}  # 保存最优的单层决策树
    bestClassEst = np.mat(np.zeros((m, 1)))  # 初始化分类后的标签数组
    minError = np.inf  # 保存最小的错误率
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 每一次划分点变化的步长
        for j in range(-1, numSteps + 1):
            for inequal in ['lt', 'gt']:  # 用来转换数据集的分类情况，例如lt时小于阀值分类为-1，gt时大于阀值才分为-1
                threshVal = (rangeMin + float(j) * stepSize)  # 设置该属性的划分点
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 返回的分类后的标签数组
                errArr = np.mat(np.ones((m, 1)))  # 初始化错误分类的数组，分类错误为1
                errArr[predictedVals == labelMatrix] = 0  # 将数据集分类正确的点标为0，该式子采用了numpy中的布尔判断
                weightedError = D.T * errArr  # 因为每个样本点都有权重，且所有的样本点的权重相加为1，因此最终的分类误差可以用分类错误点的权重相加
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['feature'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['inequal'] = inequal

    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataSet, classLabels, numIt=50):
    """
    创建弱学习器集合
    :param dataSet: 原始数据集
    :param classLabels: 原始数据集对应的标签
    :param numIt: 设定的弱学习器最大的个数
    :return:
    """
    weakClassifier = []  # 用来存储弱分类器的列表
    m = np.shape(dataSet)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化样本权重向量
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataSet, classLabels, D)
        alpha = float(0.5 * np.log((1 - error) / error))  # 计算当前的弱分类器的权重
        bestStump['alpha'] = alpha
        weakClassifier.append(bestStump)

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 获得e的指数数组，分类正确的为1，指数为-a，分类错误的为-1，指数为a
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据分类的错误点重新划分样本的权重向量

        aggClassEst += alpha * classEst  # 该分类器分类后的结果，该结果乘上了权重，便于后面的集成
        # np.sign将大于0的值标为1，小于0的值标为-1, aggErrors中分类错误的点标为1， 分类正确的点分为0
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0:  # 若在设定的最大分类器数量之前，分类器的误差就达到了0，则中止当前的循环
            break
    return weakClassifier


def adaClassify(dataToClass, weakClassifier):
    """
    利用集成学习器进行分类
    :param dataToClass:待预测的数据
    :param weakClassifier: 集成学习器
    :return:
    """
    dataMat = np.mat(dataToClass)
    m = np.shape(dataMat)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))  # 用于表示分类的结果
    for i in range(len(weakClassifier)):  # 将所有的弱学习器遍历出来
        classEst = stumpClassify(dataMat, weakClassifier[i]['feature'],
                                 weakClassifier[i]['threshVal'], weakClassifier[i]['inequal'])
        aggClassEst += weakClassifier[i]['alpha'] * classEst  # 将所有的弱分类器的分类结果叠加，最后为正则表示类别为1，为负则类别为-1
    return np.sign(aggClassEst)


def loadDataSet(filename):
    """
    将txt文件中的数据加载出来
    :param filename:
    :return:
    """
    labels = np.loadtxt(filename, usecols=(-1,))  # 取出数据集中的最后一列
    dataSet = np.loadtxt(filename)
    dataSet = np.delete(dataSet, -1, axis=1)  # 删除数据集的最后一列（最后一列是标签）
    return dataSet, labels


def errorRate(trainData, trainLabels, testData, testLabels):
    weakClassifier = adaBoostTrainDS(trainData, trainLabels)
    print(weakClassifier)
    testClass = adaClassify(testData, weakClassifier)
    m = len(testLabels)
    errArr = np.mat(np.ones((m, 1)))
    errorNum = errArr[testClass != np.mat(testLabels).T].sum()
    return errorNum / m


trainData, trainLabels = loadDataSet('../logRegress/horseColicTraining.txt')
testData, testLabels = loadDataSet('../logRegress/horseColicTest.txt')
error = errorRate(trainData, trainLabels, testData, testLabels)
print(error)