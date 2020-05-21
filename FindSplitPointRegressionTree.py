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
import CodeGenerateDecisionTree

def train_tree_regressor(data, feature_list=[]):
    print('Fill nan...')
    # 缺失值填补 训练决策树
    imp_mean = si()
    # featureList = ['nasrdw_recd_date','var_jb_28','var_jb_1']
    # featureList = ['nasrdw_recd_date']
    # x = df_train[featureList].fillna(df_train[featureList].mean()).copy()
    x = data[feature_list].fillna(-99999).copy()
    y = data.bad.copy()
    print('Train Decision Tree..')
    dtree = tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, min_samples_split=5000)
    dtree = dtree.fit(x, y)
    return x, dtree

def make_picture(x, dtree):
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

    return graph

def show_picture(graph):
    from PIL import Image
    graph.write_png(r'temp.png')
    img = Image.open(r'temp.png')
    img.show()


def main():
    # df_train, df_test = ParseData.loadPartData()
    df_train, df_test = ParseData.loadData()
    feature_list = ['nasrdw_recd_date', 'var_jb_23', 'var_jb_28']
    feature_list = ['nasrdw_recd_date', 'var_jb_43', 'var_jb_94', 'creditlimitamount_4',
                   'var_jb_15', 'var_jb_23', 'creditlimitamount_3', 'var_jb_73', 'var_jb_25', 'var_jb_22']

    x, dtree = train_tree_regressor(df_train, feature_list)
    graph = make_picture(x, dtree)
    show_picture(graph)
    train_sentence, test_sentence = CodeGenerateDecisionTree.create(dtree, feature_list)
    print(train_sentence, test_sentence)

if __name__ == '__main__':
    main()
