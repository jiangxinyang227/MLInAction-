import numpy as np
from numpy import random
import copy


def loadDataSet(fileName):
    """
    加载特征值只有2的数据值， 返回数据集和相应的类别
    :param fileName:
    :return:
    """
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    选择和i不相同的值
    :param i:
    :param m:
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    根据H和L的值对我们通过导数为0求出的a的解进行剪切得到我们想要的解
    :param aj: 利用导数为0求出的a解
    :param H: 最终a值的上界点
    :param L: 最终a值的下界点
    :return:
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataSet, classLabels, C, toler, maxIter):
    """
    SMO算法，用于求解拉格朗日乘子α的向量集和常数b
    :param dataSet: 原始数据集,类型为np.array类型
    :param classLabels: 数据集对应的分类集合
    :param C: 常数C，用来控制容错率的大小
    :param toler: 容错率
    :param maxIter: 最大迭代次数，只有在数据集上遍历了maxTier次还不发生a值的改变，则终止程序，返回a向量和b的值
    :return: 返回α向量和常数b
    """
    dataMatrix = np.mat(dataSet)  # 可以对numpy数组和python中的列表进行转换成矩阵
    labelMatrix = np.mat(classLabels).transpose()
    b = 0  # 模型函数中的常量b的初始化
    m, n = np.shape(dataMatrix)  # 获得数据集的行数和列数，分别代表这数据集的大小和特征的数量
    alphas = np.mat(random.random(m)).transpose()  # 初始化α向量，向量中的元素都是0

    iter = 0  # 用来控制迭代的循环
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # np.multiply实现了矩阵中对应的元素两两相乘, 下面的式子是f(xi) = a1y1x1xi + a2y2x2xi + ... + amymxmxi + b的描述
            fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMatrix[i])
            #  选择违反KTT条件的点来作为SMO算法中的第一个变量αi
            if ((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 返回了一个和i不相等的值j
                fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b

                Ej = fXj - float(labelMatrix[j])

                alphaIold = alphas[i].copy()  # 将老的ai复制一份保存, 为了之后进行新的a和老的a进行对比
                alphaJold = alphas[j].copy()  # 将老的aj复制一份保存

                # 求出L，H的值，便于对后面进行由导数求极小值得到的a进行剪枝处理
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaIold + alphaJold - C)
                    H = min(C, alphaJold + alphaIold)

                if L == H:  # 此时跳出循环，重新选择a的值
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
                    - dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:  # 此时跳出循环，重新选择a的值
                    continue

                alphas[j] -= labelMatrix[j] * (Ei - Ej)  # 根据推导出来的公式计算出未剪枝的aj的值
                alphas[j] = clipAlpha(alphas[j], H, L)  # 获得最终的aj的值

                if abs(alphas[j] - alphaJold) < 0.00001:  # 若更新后的aj变化不大，则跳出循环，重新选择
                    continue

                alphas[i] = alphaIold + labelMatrix[i] * labelMatrix[j] * (alphaJold - alphas[j])  # 根据aj的值求出ai的值

                b1 = b - Ei - labelMatrix[i] * dataMatrix[i, :] * dataMatrix[i, :].T * (alphas[i] - alphaIold) \
                     - labelMatrix[j] * dataMatrix[j, :] * dataMatrix[i, :].T * (alphas[j] - alphaJold)

                b2 = b - Ej - labelMatrix[i] * dataMatrix[i, :] * dataMatrix[j, :].T * (alphas[i] - alphaIold) \
                     - labelMatrix[j] * dataMatrix[j, :] * dataMatrix[j, :].T * (alphas[j] - alphaJold)

                if alphas[i] > 0 and alphas[j] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alphaPairsChanged += 1  # 记录修改了的a对的数量

        if alphaPairsChanged == 0:
            iter += 1  # 只有在for循环中途跳出时，该迭代次数才会+1，
        else:
            iter = 0  # 若发生了a对的更新成功，无论此时iter的值是多少，都重新设为0

    return b, alphas


dataSet, labels = loadDataSet('./testSet.txt')
b, alphas = smoSimple(dataSet, labels, 0.6, 0.001, 40)
print(b)
print(alphas)