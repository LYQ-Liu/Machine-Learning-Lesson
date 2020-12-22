import time
from sklearn import metrics
import pandas as pd
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #在图片中可以输出汉字

#画学习曲线的函数
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


cancer = load_breast_cancer() #载入数据
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['target'] = cancer.target

x = cancer.data
y = cancer.target

print('data:',x.shape)
print('target:',y.shape)

df.head() # 打印前五行数据

df.info() # 查看数据描述

df['target'].value_counts() # 打印数据类别及每种类别的个数

df.describe() # 查看对数值属性的概括

df.hist(bins=50,figsize=(20,15)) # 画出数据分布直方图

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)

#训练集
df_train = pd.DataFrame(x_train,columns=cancer.feature_names)
df_train['target'] = y_train
df_train

#测试集
df_test = pd.DataFrame(x_test,columns=cancer.feature_names)
df_test['target'] = y_test
df_test


start_time = time.time()
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model)
print('training took %fs!' % (time.time() - start_time))
predict = model.predict(x_test)
score = metrics.precision_score(y_test, predict)
recall = metrics.recall_score(y_test, predict)
print('precision: %.2f%%, recall: %.2f%%' % (100 * score, 100 * recall))
accuracy = metrics.accuracy_score(y_test, predict)
print('accuracy: %.2f%%' % (100 * accuracy))
c_matrix = confusion_matrix(
        y_test,  # array, Gound true (correct) target values
        predict,  # array, Estimated targets as returned by a classifier
        labels=[0, 1],  # array, List of labels to index the matrix.
        sample_weight=None  # array-like of shape = [n_samples], Optional sample weights
)
print('\nclassification_report:')
print(classification_report(y_test, predict, labels=[0, 1]))

print('\nconfusion_matrix:')
print(c_matrix)

#画学习曲线
cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
title = '决策树学习曲线'
plot_learning_curve(model, title, cancer.data, cancer.target, ylim=(0.5, 1.01), cv=cv)
curve1 = plot_roc_curve(model, x_train, y_train, alpha=0.8, name="决策树")
curve1.figure_.suptitle("乳腺癌 ROC")

# 画出决策树
dot_data = export_graphviz(model,
                            out_file=None,
                            # feature_names = iris_feature_name,
                            # class_names = iris_target_name,
                            filled=True,
                            rounded=True
                            )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("E:/python/treetwo.pdf")
