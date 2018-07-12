import numpy as np


def euclidSim(inA, inB):
    """
    返回两个向量之间的欧式距离
    :param inA:
    :param inB:
    :return:
    """
    return 1 / (1 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    """
    返回两个向量之间的皮尔逊系数
    :param inA:
    :param inB:
    :return:
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    返回两个向量之间的余弦相似度
    :param inA:
    :param inB:
    :return:
    """
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num/denom)


