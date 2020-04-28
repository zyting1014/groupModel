"""
    评估分群效果类
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串


def generateMultiImportance(model_list, strategy_name):
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    img_name_list = []
    for i, item in enumerate(model_list):
        lgb.plot_importance(model_list[i], max_num_features=10)
        plt.xlabel('子模型%d' % (i + 1))
        img_name_list.append("%s_%d" % (strategy_name, (i + 1)))
        plt.savefig('image/%s.png' % img_name_list[i], dpi=300, pad_inches=0)
    return img_name_list


def saveMultiFeatureImportance(model_list, strategy_name='default'):
    img_name_list = generateMultiImportance(model_list, strategy_name)
    r = int(len(model_list) + 1 / 2)
    c = 2

    for i, imgName in enumerate(img_name_list):
        fig = plt.subplot(len(img_name_list), 1, i + 1)
        plt.xticks([])  # 关闭刻度尺
        plt.yticks([])
        plt.axis('off')  # 关闭坐标轴
        img = imgplt.imread('image/%s.png' % (imgName))
        plt.imshow(img)
    plt.savefig('image/multiFeatureImportance_%s.png' % strategy_name, bbox_inches='tight', pad_inches=0, dpi=1200)


# 不同前k特征重要性差异 平均重合 / k
def importanceFeatureDiffer(model_list, k):
    importance_feature_list = []
    for model in model_list:
        feature_name = model.feature_name()
        importance = model.feature_importance(importance_type='split')
        feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': list(importance)})
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)

        importance_feature_list.append(list(feature_importance.head(k)['feature_name']))
    # all_feature = set([feature for one_list in importance_feature_list for feature in one_list])
    all_in_feature = importance_feature_list[0]
    for one_list in importance_feature_list:
        all_in_feature = set(all_in_feature).intersection(set(one_list))
    print('取前%d重要的特征，分群数为%d,其中%d个特征为最重要特征，占比为%f' % (k, len(model_list), len(all_in_feature), len(all_in_feature) / k))
