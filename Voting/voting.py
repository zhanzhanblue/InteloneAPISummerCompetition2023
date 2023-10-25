
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import process
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import extractfeatures
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import time
# from sklearnex import patch_sklearn, unpatch_sklearn
# unpatch_sklearn()
# from sklearnex import u
#unpatch_sklearn()

# train_data, train_labels = process.load_MNIST_train()
# test_data, test_labels = process.load_MNIST_test()
# 如果运行MNIST，将上面的注释打开就好
train_data, train_labels = process.load_HWDG_train()
test_data, test_labels = process.load_HWDG_test()
train_data = train_data.reshape(-1, 28 * 28)
# print(train_data.shape)

test_data = test_data.reshape(-1, 28 * 28)


# 处理
# 提取特征---------------------------------------在这里

def process(data):
    # 归一化
    # model = preprocessing.MinMaxScaler()
    # data = model.fit_transform(data)
    # data = preprocessing.scale(data)
    # 正则化
    # data = preprocessing.normalize(data)
    return data


def extract(data):
    new_data = []
    idx = 0
    for image in data:
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        data_ = []
        # 密度特征
        data_ += process(extractfeatures.strokes_density(image))
        # 轮廓特征
        data_ += process(extractfeatures.contour(image))
        # 投影特征
        data_ += process(extractfeatures.projection(image))
        # 重心问题
        data_ += process(extractfeatures.gravity(image))
        # 首个黑点
        data_ += process(extractfeatures.first_dot_location(image))
        # 粗网格密度
        data_ += process(extractfeatures.grid_density(image))

        # 把数字加进去
        data_.append(int(str(train_labels[idx])))
        idx += 1

        new_data.append(data_)

    new_data = np.array(new_data)
    new_data = pd.DataFrame(new_data)

    # print(new_data)
    # exit(0)

    # plt.figure(figsize=(50, 50))
    # sns.heatmap(new_data.corr(), annot=True, fmt='.5f')
    # plt.show()

    return new_data


# 此时将提取的特征运用的数据集中
# train_data = extract(train_data)
# test_data = extract(test_data)

# 提取特征，维度降得很低了 28x28

# 处理数据，正则化或归一化后的效果不是很好
train_data = process(train_data)
test_data = process(test_data)

tic1 = time.time()

# 定义权值
weight = []

# knn分类
clf1 = KNeighborsClassifier(n_neighbors=1)
clf1.fit(train_data, train_labels)
pred = clf1.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac)
print('KNN 正确率： ', ac)
# 找出最优参数
# x = []
#
# for i in range(1, 100):
#     clf1 = KNeighborsClassifier(n_neighbors=i)
#     clf1.fit(train_data, train_labels)
#     pred = clf1.predict(test_data)
#     x.append(accuracy_score(test_labels, pred, normalize=False))
#
# plt.plot(x)
# plt.show()

# 多项式朴素贝叶斯, 采用默认参数即可
clf2 = MultinomialNB()
clf2.fit(train_data, train_labels)
pred = clf2.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac)
print('多项式朴素贝叶斯正确率: ', ac)
# 找最优参数
# x = []
# y = []
# for alpha in range(1, 100):
#     clf2 = MultinomialNB(alpha=alpha / 10)
#     clf2.fit(train_data, train_labels)
#     pred = clf2.predict(test_data)
#     x.append(alpha / 10)
#     y.append(accuracy_score(test_labels, pred))
#
# plt.plot(x, y)
# plt.show()

# svc, 默认最优
clf3 = svm.SVC(probability=True)
clf3.fit(train_data, train_labels)
pred = clf3.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac * 3)
print('SVC的正确率： ', ac)

# x = []
# y = []
# for i in range(1, 20):
#     clf3 = svm.SVC(C=i/10, probability=True)
#     clf3.fit(train_data, train_labels)
#     pred = clf3.predict(test_data)
#     ac = accuracy_score(test_labels, pred, normalize=False)
#     x.append(i)
#     y.append(ac)
#
# plt.plot(x, y)
# plt.show()

# bagging
clf4 = BaggingClassifier(n_estimators=120)
clf4.fit(train_data, train_labels)
pred = clf4.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac)
print('袋装的正确率：', ac)
# x = []
# y = []
# for i in range(20, 200):
#     clf4 = BaggingClassifier(n_estimators=i)
#     clf4.fit(train_data, train_labels)
#     pred = clf4.predict(test_data)
#     ac = accuracy_score(test_labels, pred, normalize=False)
#     # print(ac)
#     x.append(i)
#     y.append(ac)
#
# plt.plot(x, y)
# plt.show()

# RandomForest

# 有模型找出最优参数
clf5 = RandomForestClassifier(n_estimators=72)
clf5.fit(train_data, train_labels)
pred = clf5.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac)
print('随机森林的正确率：', ac)

# x = []
# maxn = 0
# maxk = 0
# 找随机森林的最优参数
# for i in range(1, 100):
#     clf5 = RandomForestClassifier(n_estimators=i)
#     clf5.fit(train_data, train_labels)
#     pred = clf5.predict(test_data)
#     ac = accuracy_score(test_labels, pred, normalize=False)
#     x.append(ac)
#     if ac > maxn:
#         maxn = ac
#         maxk = i
#
# print(maxk)
# plt.plot(x)
# plt.show()

# clf5.fit(train_data, train_labels)
# pred = clf5.predict(test_data)
# print(accuracy_score(test_labels, pred, normalize=False))

# clf4.fit(train_data, train_labels)
# pred = clf4.predict(test_data)
# print(accuracy_score(test_labels, pred, normalize=False))
#
# clf3.fit(train_data, train_labels)
# pred = clf3.predict(test_data)
# print(accuracy_score(test_labels, pred, normalize=False))

# maxn = 0
# maxk = 0
# x = []

# AdaBoost
clf6 = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(splitter='random', max_features=10, max_depth=50, min_samples_split=6,
                                          min_samples_leaf=3), n_estimators=1200, learning_rate=0.005)
clf6.fit(train_data, train_labels)
pred = clf6.predict(test_data)
ac = accuracy_score(test_labels, pred, normalize=True)
weight.append(ac * 2)
print('AdaBoost的正确率： ', ac)

# XGB
clf7 = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, subsample=0.7,
                     eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
clf7.fit(train_data, train_labels)
pred = clf7.predict(test_data)
ac = accuracy_score(test_labels, pred)
weight.append(ac)
print('XGBoost正确率： ', ac)


# 用投票方式集成
# model = VotingClassifier(estimators=[('knn', clf1), ('multional', clf2), ('svc', clf3), ('bagging', clf4),
#                                      ('randomforest', clf5), ('AdaBoost', clf6)], voting='hard',
#                          weights=[20, 20, 70, 30, 33, 70])
model = VotingClassifier(estimators=[('knn', clf1), ('multional', clf2), ('svc', clf3), ('bagging', clf4),
                                     ('randomforest', clf5), ('AdaBoost', clf6), ('XGBoost', clf7)], voting='hard',
                         weights=weight)

# K折交叉验证
score = cross_val_score(estimator=model, X=train_data, y=train_labels, cv=10, n_jobs=-1, scoring='accuracy')

print('K折交叉验证结果 = ', np.mean(score))


model.fit(train_data, train_labels)

test_pred = model.predict(test_data)
# ac = accuracy_score(test_labels, test_pred, normalize=False)
# if ac > maxn:
#     maxn = ac
#     maxk = i
# x.append(accuracy_score(test_labels, test_pred, normalize=False))

# plt.plot(x)
# plt.show()
#
# print(maxk)

print('共：', len(test_pred), '个 ', test_pred)

print('Accuracy_score: ', accuracy_score(test_labels, test_pred, normalize=True))

tic2 = time.time()

print("time is :", tic2-tic1)
