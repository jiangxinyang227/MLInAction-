"""
对数据进行降维，利用pac技术，先求协方差矩阵的特征值，对特征值从大到小排序，取前k个特征值求得特征向量，利用求得的特征向量作为一组基，
利用这组基对数据进行降维转换
"""
import numpy as np


def loadDataSet(filename):
    fr = open(filename)
    dataSet = []
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    for line in stringArr:
        dataSet.append([float(item) for item in line])
    return np.mat(dataSet)


def pca(dataSet, topNfeat):
    meanVals = np.mean(dataSet, axis=0)  # 求各个特征下的平均值，为之后求协方差做准备
    meanRemoved = dataSet - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = eigVals.argsort()
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    return lowDDataMat, reconMat