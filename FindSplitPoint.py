# coding: utf-8
"""
    查找决策树切分点 图显示不出来 复制到jupyter可以显示
"""
import pandas as pd
from sklearn.impute import SimpleImputer as si
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import display,Image
from sklearn.externals.six import StringIO
import ParseData
import os

def trainTreeRegressor(data, featureList=[]):
    print('Fill nan...')
    # 缺失值填补 训练决策树
    imp_mean = si()
    # featureList = ['nasrdw_recd_date','var_jb_28','var_jb_1']
    # featureList = ['nasrdw_recd_date']
    # x = df_train[featureList].fillna(df_train[featureList].mean()).copy()
    x = data[featureList].fillna(0).copy()
    y = data.bad.copy()
    print('Train Decision Tree..')
    dtree = tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, min_samples_split=5000)
    dtree = dtree.fit(x, y)
    return x, dtree

def makePicture(x, dtree):
    print('print Tree')
    with open("dt.dot", "w") as f:
        tree.export_graphviz(dtree, out_file=f)
    dot_data = StringIO()
    tree.export_graphviz(dtree, out_file=dot_data,
                         feature_names=x.columns,
                         class_names=['bad_ind'],
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # 这是我自己的路径，注意修改你的路径
    os.environ["PATH"] += os.pathsep + 'D:/SE/graphviz-2.38/release/bin/'
    display(Image(graph.create_png()))
    plt.show()


def main():
    df_train,df_test = ParseData.loadPartData()
    x, dtree = trainTreeRegressor(df_train, ['nasrdw_recd_date'])
    makePicture(x, dtree)

if __name__ == '__main__':
    main()
