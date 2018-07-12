import numpy as np
from math import inf

def loadData(filename):
    """
    加载数据，将文件中的数据读到列表中
    :param filename:
    :return:
    """
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        curLine = [float(item) for item in curLine]
        dataMat.append(curLine)
    return np.array(dataMat)


def binSplitDataSet(dataSet, feature, value):
    """
    将数据集按照某个特征的某个划分点划分成两个数据集
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(leafData):
    """
    叶子节点数据的回归方式，在这里采用所有数据节点的平均值, 即生成叶子节点
    :return:
    """
    return np.mean(leafData[:, -1])


def regErr(subData):
    """
    返回数据集结果的总方差
    :return: np.var会返回数组中所有元素的均方差总和
    """
    return np.var(subData[:, -1]) * np.shape(subData)[0]


def linearModel(dataSet):
    """
    计算线性回归的权重系数
    :param dataSet:
    :return:
    """
    x = dataSet[:, :-1]
    y = dataSet[:, -1]
    xMat = np.mat(x)
    yMat = np.mat(y)
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx * xMat.T * yMat.T
    return ws, x, y


def modelLeaf(leafData):
    """
    叶子节点只存储线性模型的权重系数
    :param leafData:
    :return:
    """
    ws, x, y = linearModel(leafData)
    return ws


def modelErr(subData):
    if np.isnan(subData).sum() == 0:
        return 0
    ws, x, y = linearModel(subData)
    yPre = x * ws
    return np.sum(np.power(y - yPre, 2))


def chooseBestSplit(dataSet, leafType, errType, ops=None):
    """
    选择最佳的划分点和划分特征
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最小样本数，也就是说切分后的样本数少于4，则跳过从新切分
    if len(set(dataSet[:, -1].tolist())) == 1:  # 即数据集输出的结果都是一致的
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = regErr(dataSet)  # 在不进行划分时整个数据集的均方差
    bestS = np.inf  # 初始化均方差
    bestFeature = 0  # 初始化特征
    bestValue = 0  # 初始化划分点

    for featIndex in range(n-1):
        for value in dataSet[:, featIndex]:  # 直接用特征的值进行划分，通过遍历比较均方差来找到该特征下最优的解
            subData1, subData2 = binSplitDataSet(dataSet, featIndex, value)
            if (np.shape(subData1)[0] < tolN) or (np.shape(subData2)[0] < tolN):  # 切分后的样本数小于最小样本数，则跳过
                continue
            meanError = regErr(subData1) + regErr(subData2)
            if meanError < bestS:  # 如果当前的均方差小于最小均方差，则将当前的参数设为最优参数
                bestS = meanError
                bestFeature = featIndex
                bestValue = value

    if (S - bestS) < tolS:  # 如果对当前数据集进行划分后的方差较划分前的方差降低的不明显，则直接返回为叶子节点
        return None, leafType(dataSet)

    subData1, subData2 = binSplitDataSet(dataSet, bestFeature, bestValue)
    if (np.shape(subData1)[0] < tolN) or (np.shape(subData2)[0] < tolN):  # 最优参数切分后存在某一个样本集的样本数小于最小样本数，则直接返回为叶子节点
        return None, leafType(dataSet)

    return bestFeature, bestValue


def createTree(dataSet, leafType, errType, ops=(1, 4)):
    """
    创建回归树，采用递归的形式，每个内部节点保存的数据都是spVal，spInd，left子节点，right子节点
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops: 其实在这里设置的参数相当于对决策树进行了预剪枝处理，但是预剪枝往往是很难得到最优树，甚至有可能造成欠拟合或者没有去除过拟合
    :return:{'spInd': 1, 'spVal': 0.39435,
            'left': {'spInd':, 'spVal':, 'left': {'spInd':, 'spVal':, 'left':, 'right':},'right': },
            'right': {'spInd':, 'spVal':, 'left': 1.0289583666666666, 'right': -0.023838155555555553}}
    """
    feature, value = chooseBestSplit(dataSet, leafType, errType, ops)
    if feature is None:  # 递归的终止条件，主要还是由chooseBestSplit函数来控制的
        return value
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = value
    lSet, rSet = binSplitDataSet(dataSet, feature, value)
    retTree['left'] = createTree(lSet, leafType, errType)
    retTree['right'] = createTree(rSet, leafType, errType)
    return retTree


def isTree(obj):
    """
    判断当前对象是否是树
    :param obj:
    :return:
    """
    return isinstance(obj, dict)


def getMean(tree):
    """
    将所有的叶子节点自下往上坍塌，最终只返回一个根节点
    :param tree:
    :return:
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    对决策回归树进行剪枝，决策树是一颗完满二叉树，所有的非叶子节点的度都是2
    该方法的核心思想是从底部开始合并叶子节点，看是否会降低测试误差，在判断的过程中只是局部判断，数据集也是分到了该子树的数据集，
    然而局部的优化不代表全局的优化，利用各个局部的优化来叠加不一定能获得最优子树，当然这种方法计算和理解都比较简单，
    配合预剪枝一起使用也会获得较好的结果
    :param tree:
    :param testData:
    :return:
    """
    if np.shape(testData)[0] == 0:  # 当测试集为空时，直接返回根节点，根节点的值是根据各个叶子节点坍塌后的值
        return getMean(tree)

    if isTree(tree['left']) or isTree(tree['right']):  # 若当前节点的子节点至少有一个不为叶子节点，则递归下去
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])  # 让测试集按照当前树的参数进行划分
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['left']) and not isTree(tree['right']):  # 若当前节点只有叶子节点，则判断叶子节点合并后的测试误差是否有变小
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = modelErr(lSet) + modelErr(rSet)
        treeMean, x, y = linearModel(testData)  # 对叶子节点进行合并，并将合并的值返给上一节点
        errorMergr = modelErr(testData)
        if errorMergr < errorNoMerge:
            return treeMean
        else:
            return tree

    return tree


dataSet = loadData('./exp2.txt')

regTree = createTree(dataSet[:-20, :], modelLeaf, modelErr)
print(regTree)
tree = prune(regTree, dataSet[-20:, :])

print(tree)