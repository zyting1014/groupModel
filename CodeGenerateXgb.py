"""
    通过解析graphviz.files.Source，生成xgboost的划分代码
"""

import Node
import re
import sys
from queue import Queue

nodes = []
for i in range(500):
    nodes.append(Node.Node(-1))


def generate_tree_structure(img):
    print('in %s' % sys._getframe().f_code.co_name)
    info = img.source
    info = info.split('\n\n')
    for three_item in info[1:-1]:
        if three_item.find('leaf') != -1:  # 叶子节点 跳过
            continue
        three_item = three_item.split('\n')  # 每条有三行
        position = int(re.search('(\d+)', three_item[0]).group(1))
        split = re.search('label="(.*)" ]', three_item[0]).group(1)
        left_position = int(re.search('-> (.*) \[', three_item[1]).group(1))
        right_position = int(re.search('-> (.*) \[', three_item[2]).group(1))
        print('position:%s,split:%s,left_position:%s,right_position:%s' % (
            position, split, left_position, right_position))
        nodes[position].value = split
        nodes[position].left = left_position
        nodes[position].right = right_position


def level_tranverse():
    q = Queue()
    q.put(0)
    res = []

    def tranverse():
        if q.empty(): return
        n = q.get()
        if nodes[n].value == -1: return
        res.append(nodes[n].value)
        q.put(nodes[n].left)
        q.put(nodes[n].right)
        tranverse()

    tranverse()
    return res


def generate_sentance(res):
    sentence = 'df_train, column_name = GroupFunc.decisionTreeMethod3(df_train'
    for item in res[:7]:
        item = item.split('<')
        sentence += ', "'
        sentence += item[0]
        sentence += '", '
        sentence += item[1]
    sentence += ')'
    train_sentence = sentence
    test_sentence = sentence.replace('train', 'test')
    return train_sentence, test_sentence


def create(img):
    generate_tree_structure(img)
    res = level_tranverse()
    train_sentence, test_sentence = generate_sentance(res)
    return train_sentence, test_sentence
