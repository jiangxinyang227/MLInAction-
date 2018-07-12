import numpy as np


def loadData(filename):
    """
    从txt文件中将数据读出来，保存成np的数组形式
    :param filename:
    :return:
    """
    dataSet = []
    fr = open(filename)
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        dataSet.append([float(item) for item in lineList])
    return np.array(dataSet)


def calcuDistance(vec1, vec2):
    """
    计算两个向量之间的欧氏距离
    :param vec1:
    :param vec2:
    :return:
    """
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


def randomCenter(dataSet, k):
    """
    初始化聚类的簇心, 让初始化的簇心的每个特征上的值都在数据集特征值的范围内
    :param dataSet:
    :param k: 聚类的簇心个数
    :return: 返回k个簇心组成的数组
    """
    n = np.shape(dataSet)[1]
    centers = np.mat(np.zeros((k, n)))
    for i in range(n):
        minValue = dataSet[:, i].min()
        rangeValue = float(dataSet[:, i].max() - minValue)
        minValue = np.tile(minValue, (k, 1))
        centers[:, i] = minValue + rangeValue * np.random.rand(k, 1)  # random.rand随机生成的值的在0-1范围内
    return centers


def k_means(dataSet, k, distance, center):
    """
    k-means算法
    :param dataSet: 原始数据集
    :param k: 聚类的簇心
    :param distance: 计算向量之间的距离的方法
    :param center: 初始化簇心的方法
    :return:
    """
    centers = np.array(center(dataSet, k))
    m = np.shape(dataSet)[0]
    clusterAssment = np.zeros((m, 2))  # 用来存储数据的下标和对应的类别
    centerChanged = True  # 判断是否需要继续迭代更改簇心进行聚类
    while centerChanged:
        centerChanged = False
        # 先将各数据进行聚类
        for i in range(m):
            minDist = np.inf; minIndex = -1  # 用来存储当前数据和簇心的最小距离和簇心的下标
            for j in range(k):
                dist = distance(dataSet[i, :], centers[j, :])

                if dist < minDist:
                    minDist = dist; minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 如果还是存在数据点的簇心在本次聚类过程中发生了改变，则继续迭代
                centerChanged = True

            clusterAssment[i, :] = minIndex, minDist ** 2  # 将当前的类别和与簇心之间的最小距离保存

        # 更新簇心的向量
        for i in range(k):
            subData = dataSet[clusterAssment[:, 0] == i, :]  # 从数据集中取出类别为i的数据
            centers[i, :] = np.mean(subData, axis=0)  # 对当前簇心下所有的数据取均值作为当前的簇心

    return clusterAssment, centers


def binaryKmeans(dataSet, k, distance):
    """
    二分聚类一开始将整个数据集看做是一个簇，然后对其进行二分，即每次只增加一个簇，直到簇的个数达到了k个，在选择某一簇进行二分类的时候，
    是采用对所有的簇进行二分类，取军方误差最小的那一簇进行二分类
    :param dataSet:
    :param k:
    :param distance:
    :return:
    """
    m = np.shape(dataSet)[0]
    center0 = np.mean(dataSet, axis=0)  # 初始化第一个簇心
    centers = np.array([center0])
    clusterAssment = np.zeros((m, 2))
    for i in range(m):
        clusterAssment[i, 1] = distance(dataSet[i, :], center0) ** 2  # 将初始时的各数据的均方差保存在数组中
    while len(centers) < k:
        minSSE = np.inf
        bestCentToSplit = -1  # 记录最优的二分聚类的簇
        bestNewCenter = []  # 记录二分聚类后返回的簇心

        for j in range(len(centers)):
            subData = dataSet[clusterAssment[:, 0] == j, :]  # 将当前簇心下的数据都取出来
            classifier, centerK = k_means(subData, 2, distance, randomCenter)
            subSSE = np.sum(classifier[:, 1])  # 当前簇进行簇心为2的聚类后的误差
            extraSubSSE = np.sum(clusterAssment[clusterAssment[:, 0] != j, 1])  # 将非当前簇心的样本均方差相加
            totalSSE = subSSE + extraSubSSE  # 对某一簇进行二分聚类之后的误差
            if totalSSE < minSSE:
                minSSE = totalSSE
                bestCentToSplit = j
                bestNewCenter = centerK
                bestClassAssment = classifier.copy()  # 将返回的二分类的数据复制一份，对应的类别是0,1, 在这里我们需要将类别重新赋值，然后放入到原始的数据集类别中

        bestClassAssment[bestClassAssment[:, 0] == 1, 0] = len(centers)  # 将二分类后的数据中的1类归为centers的长度
        bestClassAssment[bestClassAssment[:, 0] == 0, 0] = bestCentToSplit  # 将二分类后的数据中的0类归为被二分之前的簇类
        centers[bestCentToSplit] = bestNewCenter[0]
        centers = np.row_stack((centers, bestNewCenter[1]))
        clusterAssment[clusterAssment[:, 0] == bestCentToSplit] = bestClassAssment
    return clusterAssment, centers


dataSet = loadData('./testSet.txt')
clusters, centers = binaryKmeans(dataSet, 4, calcuDistance)
print(clusters)
print(centers)