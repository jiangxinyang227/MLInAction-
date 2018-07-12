
def loadData(filename):
    fr = open(filename)
    dataSet = []
    for line in fr.readlines():
        curList = line.strip().split(' ')

        dataSet.append(curList)
    return dataSet


def createC1(dataSet):
    """
    返回数据集中单个元素集合的总不可变集合
    :param dataSet:
    :return:
    """
    c1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()  # 对列表中的各元素进行排序
    # 在Python中set是可变集合，frozenset是不可变集合，存在哈希值
    return map(frozenset, c1)  # map(func, iteration),对后面的可迭代对象中的所有元素代入到func中，并将返回的结果按原索引位置组成


def scanD(dataSet, ck, minSupport):
    """
    用于返回频繁项集和最频繁项集的支持度
    :param dataSet: 原始数据集
    :param ck: 候选集，即要进行计算支持度的候选集合，如c1， c2
    :param minSupport: 最小支持度
    :return:
    """
    cDict = {}
    # 将ck集合中各个集合出现的次数统计出来
    for item in ck:
        for data in dataSet:
            if item.issubset(data):
                cDict[item] = cDict.get(item, 0) + 1

    total = len(dataSet)  # 数据集的总数，用于计算支持度
    retList = []  # 用来存储频繁项集
    supportData = {}  # 用来存储频繁项集的支持度

    for key in cDict.keys():
        support = cDict[key] / total
        if support > minSupport:
            retList.append(key)
            supportData[key] = support

    return retList, supportData


def aprioriGen(Lk, k):
    """
    用来处理在上一个频繁项集的基础上获得当前的候选子集
    :param Lk: 上一个频繁项集
    :param k: k的值其实是上一个频繁项集中元素的长度 + 1的值，很精妙的设计
    :return:
    """
    retList = []  # 用来存储当前的候选集
    lenLk = len(Lk)
    # 用两层for循环来合并频繁项集中的各元素，但是要避免重复
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]  # 例如在进行长度为1的频繁项合并成2时，L1为空，所有的都要去两两合并
            L2 = list(Lk[j])[:k-2]  # 而让长度为2的合并成2时，只要判断第一个值是否相等，因为第一个值相等，第二个值必然不等，合并成的3中的各元素也必然不等，
            if L1 == L2:            # 而且最精妙的是也避免可在2合并成3时会出现3的重复
                retList.append(Lk[i] | Lk[j])  # | 是对集合取并集
    return retList


def apriori(dataSet, minSupport=0.7):
    """
    1，先把c1生成，通过scanD函数获得c1的频繁项集L1和对应的支持度
    2，借助aprioriGen函数在c1的基础上生成更多的ck，通过scanD函数生成对应的频繁项集和支持度
    3，当最终某一时刻导入的ci生成的频繁项集为0时，终止整个过程
    :param dataSet: 原始数据集
    :param minSupport: 最小的支持度
    :return:
    """
    c1 = createC1(dataSet)  # 将单个元素组成集合的列表
    D = dataSet  # 函数式编程，直接将dataSet中的数据映射成集合的形式
    L1, supportData = scanD(dataSet, c1, minSupport)  # 获得c1中的频繁项集和相应的支持度
    L = [L1]  # 用来保存每次生成的频繁项集，每一个频繁项集列表总的频繁项集中的元素个数是不一致的
    k = 2  # 设置k值，可以用来获得当前所需要导入到aprioriGen函数中去获得新的ck时的频繁项集列表
    while len(L[k-2]) > 0:  # 循环终止条件就是当前的频繁项集列表为空
        ck = aprioriGen(L[k-2], k)  # 根据频繁项集获得ck
        Lk, supk = scanD(D, ck, minSupport)  # 获得ck对应的Lk和支持度
        supportData.update(supk)  # 将频繁项集的支持度更新到supportData中
        L.append(Lk)  # 将当前的频繁项集列表加入到L中
        k += 1
    return L, supportData


def generateRules(L, supportDatam, minConf=0.7):
    """
    生成满足可信度的规则
    :param L: apriori函数返回的第一个值，各频繁项集的列表
    :param supportDatam: 各频繁项的支持度
    :param minConf: 最小的可信度
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):  # 当频繁项的长度是1时就不需要进行关联规则分析，因此从第一项开始，即关联项的长度必须为2
        for fregSet in L[i]:  # 将当前关联项集中的关联项取出来
            # 将关联项中的各元素取出来，并且转成集合的形式保存起来 例如[frozenset({2}), frozenset({3})]，
            # 这也是为了后面计算时可以直接在supportData中拿到相应的支持度
            H1 = [frozenset([item]) for item in fregSet]
            if i > 1:
                print(H1)
                rulesFromConseq(fregSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(fregSet, H1, supportData, bigRuleList, minConf)  # 当频繁项中的元素只要两个时，可以直接计算可信度
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf):
    """
    计算各种关联规则下的可信度，并保存关联规则和相对应的可信度
    :param freqSet: 要计算关联规则的频繁项
    :param H: 关联规则中后边的元素的集合组成的列表
    :param supportData: 频繁项和对应的支持度的字典
    :param br1: 用来存储关联规则和相应的可信度，因为传入的是列表，因此在该函数中添加了值，在主函数中也会有相应的改变
    :param minConf: 最小的可信度
    :return:
    """
    prunedH = []  # 用来存储关联规则中的右侧元素
    for conseq in H:
        # 计算可信度，如{尿布，葡萄酒}，{尿布}，则尿布->葡萄酒={尿布，葡萄酒}/{尿布} “-”表示集合中的差集
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # freqSet-conseq表示左项，conseq表示右项
            # print(freqSet-conseq, "---->", conseq, "conf:", conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minconf):
    """
    当频繁项中的元素大于2时，会调用该函数
    :param freqSet: 要计算关联规则的频繁项
    :param H: # 频繁项中的各元素组成的列表
    :param supportData: 频繁想对应的支持度的字典
    :param br1: 保存关联规则和可信度
    :param minconf:最小的可信度
    :return:
    """
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m + 1)  # 用来获得候选子集，也就是关联分析的右侧项
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minconf)  # 计算当前的可信度
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minconf)  # 直到候选子集的个数少于1时停止合并右侧项


dataSet = loadData('./mushroom.dat')
c1 = createC1(dataSet)
retList, supportData = apriori(dataSet)
bigRuleList = generateRules(retList, supportData)
# print(retList)
# print(supportData)
# print(bigRuleList)