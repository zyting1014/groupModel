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
    print('评估：进入特征重要性绘图函数..')
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
def importanceFeatureDiffer(model_list, k=20):
    print('评估：进入不同前k特征重要性差异函数..')
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


# 与单模型woe不同 分群woe是将所有数据混在一起 找到最优分箱点 再分开来计算woe
# 传入划分完毕的数据集合 和绘制曲线的重要特征名称
def multiWoe(group_list, feature_name, n=10):
    print('评估：进入woe曲线绘制函数..')
    def drawWoeCurve():
        # 绘制woe曲线
        plt.title('woe curve')
        for item in woe_list:
            x = range(1, len(woe_list[0]) + 1)
            plt.plot(x, item)
        plt.xlabel('%s等频分箱占比' % feature_name)
        plt.ylabel('woe值')
        plt.show()
        plt.savefig('%s等频分箱占比.png' % feature_name)

    # 等频分箱 这里没有卡方分箱 因为不用那么麻烦 而且很多变量也分不出来 数据量太大 卡方是协方差制类的 就看个大概趋势
    group_list2 = group_list.copy()
    print('正在整合数据...')
    for i, group in enumerate(group_list2):
        group['model_num'] = i
    flatten_data = group_list[0]
    for i in range(1, len(group_list)):
        flatten_data = pd.concat([flatten_data, group_list[i]], axis=0)

    print('正在进行等频分箱...')
    flatten_data['bin_num'] = pd.qcut(flatten_data[feature_name], n)

    woe_list = []
    each_model = flatten_data.groupby(['model_num'], as_index=True)
    print('正在逐群体逐段计算woe值..')
    for i, data in each_model:
        print(data)
        good = data.bad.sum()
        bad = data.bad.count() - good
        group = data.groupby('bin_num', as_index=True)
        each_woe = []
        for j, each_group in group:
            rate = each_group.bad.mean()
            woe = np.log((rate / good) / ((1 - rate) / bad))
            each_woe.append(woe)
        woe_list.append(each_woe)
    print('正在绘制woe曲线...')
    drawWoeCurve()


def main(model_list):
    saveMultiFeatureImportance(model_list, 'test')
    importanceFeatureDiffer(model_list, 10)
    importanceFeatureDiffer(model_list, 20)
