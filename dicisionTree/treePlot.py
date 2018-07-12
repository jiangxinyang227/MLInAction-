import matplotlib.pyplot as plt
from trees import myTree


decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 定义判断节点的形态
leafNode = dict(boxstyle="round4", fc="0.8")  # 定义叶子节点的形态
arrow_args = dict(arrowstyle="<-")  # 定义箭头


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解
    :param nodeTxt: 节点的文字标注，决策节点为属性名，叶子节点为类别名
    :param centerPt: 节点中心位置
    :param parentPt: 箭头起始位置（上一节点位置）
    :param nodeType: 节点属性，decisionNode or leafNode
    :return:
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction", xytext=centerPt,
                            textcoords="axes fraction", va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot(inTree):
    """
    创建绘图区
    :return:
    """
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def getNumLeafs(myTree):
    """
    获得树的叶子节点个数，便于之后x轴的坐标的划分
    :param myTree:
    :param numLeafs:
    :return:
    """
    numLeafs = 0
    firstStr = [key for key in myTree.keys()][0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    得到树的深度，便于以后y轴的坐标划分
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = [key for key in myTree.keys()][0]  # 拿到第一个属性节点
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth

    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    :param cntrPt: 子节点位置
    :param parentPt: 父节点位置
    :param txtString: 标注内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注的x坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 计算标注的y坐标
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = [key for key in myTree.keys()][0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


createPlot(myTree)