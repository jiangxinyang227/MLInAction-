"""
本文主要是数据的分割，归一化和交叉验证处理
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn import svm


iris = datasets.load_iris()
x = iris.data
y = iris.target

# 对样本集进行分割
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.4)
clf = svm.SVC(kernel='linear', C=1).fit(xTrain, yTrain)
score = clf.score(xTest, yTest)


from sklearn.model_selection import cross_val_score
clf1 = svm.SVC(kernel='linear', C=1)
# 采用交叉验证，直接自动分割数据，并返回验证的结果，只要导入模型，数据集，设置交叉验证的次数cv=k
scores = cross_val_score(clf1, x, y, cv=5)


from sklearn.model_selection import ShuffleSplit
n_samples = np.shape(iris.data)[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores1 = cross_val_score(clf1, x, y, cv=cv)


"""
对训练数据和测试数据做归一化处理
"""
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(xTrain)  # 根据训练数据初始化归一化模型
xTrainNorm = scaler.transform(xTrain)
xTestNorm = scaler.transform(xTest)
clf = svm.SVC(kernel='linear', C=1).fit(xTrainNorm, yTrain)
score1 = clf.score(xTestNorm, yTest)


# 采用归一化加交叉验证来处理
scaler1 = preprocessing.StandardScaler().fit(x)
xNorm = scaler1.transform(x)
scores2 = cross_val_score(clf1, xNorm, y, cv=5)


from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf1, xNorm, y, cv=10)  # 返回所有交叉验证的测试集的类别结果
score3 = metrics.accuracy_score(y, predicted)  # 根据类别结果和真实值进行对比，获得误差，返回的结果相当于是一个均值


from sklearn.metrics import mean_absolute_error  # 平均绝对误差
from sklearn.metrics import mean_squared_error  # j均方误差
from sklearn.metrics import r2_score  # r2分数，确定系数，用来判断回归方程的拟合程度


