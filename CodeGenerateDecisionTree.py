"""
    通过解析sklearn.tree.tree.DecisionTreeRegressor，生成决策树的划分代码
"""

import Node
import re
import sys
from queue import Queue
import CodeGenerateXgb

nodes = []
for i in range(500):
    nodes.append(Node.Node(-1))


def get_attribute(dtree):
    feature = dtree.tree_.feature
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    threshold = list(dtree.tree_.threshold)
    return feature, children_left, children_right, threshold


def generate_tree_structure(feature_list, feature, children_left, children_right, threshold):
    for i in range(len(feature_list)):
        left = children_left[i]
        right = children_right[i]
        num = threshold[i]
        if left == right:
            continue
        position = i
        split = feature_list[feature[i]]

        nodes[position].value = split + '<' + str(num)
        nodes[position].left = left
        nodes[position].right = right


def create(dtree, feature_list):
    feature, children_left, children_right, threshold = get_attribute(dtree)
    generate_tree_structure(feature_list, feature, children_left, children_right, threshold)
    res = CodeGenerateXgb.level_tranverse()
    train_sentence, test_sentence = CodeGenerateXgb.generate_sentance(res)
