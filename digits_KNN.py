import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#数据加载
digits = load_digits()
#在这里因为数据集存储的是数组，不是dataframe，因此要了解数据的基本情况可以直接在变量集中查看
data = digits.data
#数据初步探索
print(data.shape)
print(digits.images[0])
print(digits.target[0])
#显示出第一幅图像
plt.gray()  #显示灰度图，这个在这里可有可无
plt.imshow(digits.images[0])
plt.show()

target = digits.target
#将数据集分割为测试集和训练集
train_x,test_x,train_target,test_target = train_test_split(data,target,test_size = 0.25)
#规范化Z-Score方法
ss = StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.fit_transform(test_x)
#创建KNN分类器并进行训练
knn = KNeighborsClassifier()
knn.fit(train_ss_x,train_target)
#进行手写数字的预测并对模型预测准确率进行评估
predict_target = knn.predict(test_ss_x)
acc_score = accuracy_score(predict_target,test_target)
print("KNN分类树预测准确率：",acc_score)
#KNN分类树预测准确率： 0.9613   test_size = 0.33
#KNN分类树预测准确率： 0.98     test_size = 0.25    
#KNN分类树预测准确率： 0.86     test_size = 0.25   n_neighbors=200 
#KNN分类树预测准确率： 0.9822   test_size = 0.25   n_neighbors=2
#KNN分类树预测准确率： 0.98      test_size = 0.25   n_neighbors=10

#用于对比不同分类器的预测准确率
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

#SVM分类器
svm = SVC()
svm.fit(train_ss_x,train_target)
predict_target_svm = svm.predict(test_ss_x)
print("SVM分类准确率：",accuracy_score(predict_target_svm,test_target))

#Naive Bayes分类器
'''
在做多项式朴素贝叶斯分类的时候传入的数据不能有负数。
Z-Score会将数值规范化维标准正态分布包含负数，因此用
Min-Max规范化，将数据规范化到（0-1）内
'''
#MinMax规范化
mm = MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.fit_transform(test_x)

mnb = MultinomialNB()
mnb.fit(train_mm_x,train_target)
predict_target_mnb = mnb.predict(test_mm_x)
print("多项式朴素贝叶斯分类准确率为：",accuracy_score(predict_target_mnb,test_target))

#CART决策树分类器
clf = DecisionTreeClassifier()
clf.fit(train_mm_x,train_target)
predict_target_clf = clf.predict(test_mm_x)
print("CART决策树分类器预测准确率为：",accuracy_score(predict_target_clf,test_target))

#SVM分类准确率： 0.9888888888888889
#多项式朴素贝叶斯分类准确率为： 0.8955555555555555
#CART决策树分类器预测准确率为： 0.8711111111111111


















