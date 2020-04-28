"""
    分群作为新特征函数 被StandardVersion类调用
"""
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import numpy as np
import pandas as pd
import os


# 决策树划分1
def decisionTreeMethod1(data_origin):
    print('in decisionTreeMethod1..')
    column_name = []
    for i in range(6):
        column_name.append('seg%d' % (i + 1))
    data = data_origin.copy(deep=True)
    feature_list = ['var_jb_23', 'var_jb_28', 'nasrdw_recd_date']
    data = data[feature_list].fillna(-99999)

    data['seg1'] = 0
    data['seg2'] = 0
    data['seg3'] = 0
    data['seg4'] = 0
    data['seg5'] = 0
    data['seg6'] = 0

    data_origin.loc[
        (data['var_jb_28'] <= 4.5) & (data['var_jb_23'] <= 27.5) & (data['nasrdw_recd_date'] <= 20181023), 'seg1'] = 1
    data_origin.loc[
        (data['var_jb_28'] <= 4.5) & (data['var_jb_23'] <= 27.5) & (data['nasrdw_recd_date'] > 20181023), 'seg2'] = 1
    data_origin.loc[
        (data['var_jb_28'] <= 4.5) & (data['var_jb_23'] > 27.5) & (data['nasrdw_recd_date'] <= 20181023), 'seg3'] = 1
    data_origin.loc[
        (data['var_jb_28'] <= 4.5) & (data['var_jb_23'] > 27.5) & (data['nasrdw_recd_date'] > 20181023), 'seg4'] = 1
    data_origin.loc[(data['var_jb_28'] > 4.5) & (data['nasrdw_recd_date'] <= 20181011), 'seg5'] = 1
    data_origin.loc[(data['var_jb_28'] > 4.5) & (data['nasrdw_recd_date'] > 20181011), 'seg6'] = 1

    return data_origin, column_name


# 决策树划分2
def decisionTreeMethod2(data_origin):
    print('in decisionTreeMethod2..')
    column_name = []
    for i in range(12):
        column_name.append('seg%d' % (i + 1))
    data = data_origin.copy(deep=True)
    featureList = ['nasrdw_recd_date', 'var_jb_23', 'creditlimitamount_4']
    data[featureList].fillna(-99999)

    data['seg1'] = 0
    data['seg2'] = 0
    data['seg3'] = 0
    data['seg4'] = 0
    data['seg5'] = 0
    data['seg6'] = 0
    data['seg7'] = 0
    data['seg8'] = 0
    data['seg9'] = 0
    data['seg10'] = 0
    data['seg11'] = 0
    data['seg12'] = 0

    data_origin.loc[
        (data['creditlimitamount_4'] <= 299.5), 'seg1'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] <= 65954.0) & (data['creditlimitamount_4'] > 299.5), 'seg2'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 65954.0) & (data['creditlimitamount_4'] <= 329720.0), 'seg3'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 329720.0), 'seg4'] = 1

    ##########################
    data_origin.loc[
        (data['creditlimitamount_4'] <= 299.5) & (data['var_jb_23'] <= 10.5), 'seg5'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] <= 299.5) & (data['var_jb_23'] > 10.5), 'seg6'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] <= 65954.0) & (data['creditlimitamount_4'] > 299.5) & (
                data['nasrdw_recd_date'] <= 20181023.0), 'seg7'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] <= 65954.0) & (data['creditlimitamount_4'] > 299.5) & (
                data['nasrdw_recd_date'] > 20181023.0), 'seg8'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 65954.0) & (data['creditlimitamount_4'] <= 329720.0) & (
                data['nasrdw_recd_date'] <= 20181023.0), 'seg9'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 65954.0) & (data['creditlimitamount_4'] <= 329720.0) & (
                data['nasrdw_recd_date'] > 20181023.0), 'seg10'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 329720.0) & (data['creditlimitamount_4'] <= 509500.0), 'seg11'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 329720.0) & (data['creditlimitamount_4'] > 509500.0), 'seg12'] = 1

    return data_origin, column_name


def decisionTreeMethod3(data, n1, s1, c1, cs1, c2, cs2):
    print('in decisionTreeMethod3..')
    column_name = []
    for i in range(4):
        column_name.append('seg%d' % (i + 1))

    data['seg1'] = 0
    data['seg2'] = 0
    data['seg3'] = 0
    data['seg4'] = 0

    data.loc[((data[n1] < s1) | (data[n1].isnull())) & ((data[c1] < cs1) | (data[c1].isnull())), 'seg1'] = 1
    data.loc[((data[n1] < s1) | (data[n1].isnull())) & (data[c1] >= cs1), 'seg2'] = 1
    data.loc[(data[n1] >= s1) & ((data[c2] < cs2) | (data[c2].isnull())), 'seg3'] = 1
    data.loc[(data[n1] >= s1) & (data[c2] >= cs2), 'seg4'] = 1

    return data, column_name


# 高斯混合模型 用所有变量聚类
def getGMMCategoryFeature(df_train, df_test, feature_categorical, n_components=4):
    print('in getGMMCategoryFeature..')
    import StandardVersion
    print(feature_categorical)

    li = ["nasrdw_recd_date", "var_jb_20", "var_jb_21", "var_jb_24", "var_jb_28", "var_jb_33", "var_jb_34", "var_jb_40",
          "var_jb_75", "var_jb_76", "var_jb_79", "var_jb_88", "var_jb_89", "var_jb_91", "var_jb_92",
          "curroverdueamount_3", "creditlimitamount_4", "balance", "remainpaymentcyc", "scheduledpaymentamount_4",
          "actualpaymentamount_4", "curroverduecyc_4", "curroverdueamount_4", "overdue31to60amount_7",
          "overdue61to90amount_7", "overdue91to180amount_7", "creditlimitamount", "overdueover180amount_8",
          "latest5yearoverduebeginmonth", "latest5yearoverdueendmonth", "var_jb_4_O", "guaranteetype_3_1|质押（含保证金）",
          "guaranteetype_3_3|保证", "currency_3_澳门元", "financetype_3_住房储蓄银行", "financetype_3_外资银行", "paymentrating_02|周",
          "guaranteetype_4_4|信用/免担保", "type_91|个人消费贷款", "financetype_4_住房公积金管理中心", "financetype_4_消费金融有限公司",
          "class5state_5|损失", "var_jb_48_专升本", "var_jb_48_博士研究生", "var_jb_48_夜大电大函大普通班", "var_jb_55_不详",
          "var_jb_55_夜大学", "var_jb_47_全日制", "var_jb_47_夜大学", "var_jb_47_研究生", "var_jb_54_全日制", "var_jb_54_夜大学",
          "var_jb_54_研究生", "guaranteetype_1|质押（含保证金）", "guaranteetype_6|组合（不含保证）", "state_3|止付", "state_5|呆帐",
          "var_jb_39_B", "var_jb_39_D", "var_jb_39_K", "var_jb_39_L", "var_jb_93_Z"]
    gmm_list = feature_categorical + li
    print('特征数量：%d' % (len(gmm_list)))

    x_train = df_train[gmm_list].copy().fillna(-99999)
    x_test = df_test[gmm_list].copy().fillna(-99999)
    x_train = StandardVersion.proprocessCateory(x_train, feature_categorical)
    x_test = StandardVersion.proprocessCateory(x_test, feature_categorical)

    feature_num = x_train.shape[1]

    if os.path.exists('model/GMMCategoryFeature%d_%d.model' % (n_components, feature_num)):
        print('加载GMMCategoryFeature文件..')
        gmm = StandardVersion.loadModel('GMMCategoryFeature%d_%d.model' % (n_components, feature_num))
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(x_train)  # 可以调reg_covar=0.0001
        print('训练完毕')
        StandardVersion.saveModel(gmm, 'GMMCategoryFeature%d_%d.model' % (n_components, feature_num))

    labels_train = gmm.predict(x_train)
    labels_test = gmm.predict(x_test)

    df_train['gmm'] = labels_train.tolist()
    df_test['gmm'] = labels_test.tolist()

    # 转为one-hot编码
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['gmm'], 'GMMCategoryFeature')
    # df_train = df_train.drop('gmm', axis=1)
    # df_test = df_test.drop('gmm', axis=1)
    return df_train, df_test, column_name


def getGMMNullFeature(df_train, df_test, n_components=4):
    print('in getGMMNullFeature..')
    import StandardVersion
    X_train = df_train.copy()
    X_test = df_test.copy()

    # 空值为1 非空为0 降维
    df_train_null = X_train.where(X_train.isnull(), 0).fillna(1).astype(int)
    df_test_null = X_test.where(X_test.isnull(), 0).fillna(1).astype(int)
    pca = PCA(n_components=10)
    df_train_null = pca.fit_transform(df_train_null.values)
    df_test_null = pca.transform(df_test_null.values)
    print(df_train_null.shape)
    print(df_test_null.shape)

    if os.path.exists('model/GMMNullFeature%d.model' % n_components):
        print('加载GMMCategoryFeature文件..')
        gmm = StandardVersion.loadModel('GMMNullFeature%d.model' % n_components)
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(df_train_null)

        print('训练完毕')
        StandardVersion.saveModel(gmm, 'GMMNullFeature%d.model' % n_components)

    labels_train = gmm.predict(df_train_null)
    labels_test = gmm.predict(df_test_null)
    df_train['gmmNull'] = labels_train.tolist()
    df_test['gmmNull'] = labels_test.tolist()

    print(df_train['gmmNull'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['gmmNull'], 'GMMNullFeature')
    return df_train, df_test, column_name


def getKmeansNullFeature(df_train, df_test, n_components=4):
    print('in getKmeansNullFeature..')
    import StandardVersion
    X_train = df_train.copy()
    X_test = df_test.copy()

    # 空值为1 非空为0 降维
    df_train_null = X_train.where(X_train.isnull(), 0).fillna(1).astype(int)
    df_test_null = X_test.where(X_test.isnull(), 0).fillna(1).astype(int)
    pca = PCA(n_components=10)
    df_train_null = pca.fit_transform(df_train_null.values)
    df_test_null = pca.transform(df_test_null.values)
    print(df_train_null.shape)
    print(df_test_null.shape)

    if os.path.exists('model/KmeansNullFeature%d.model' % n_components):
        print('加载KmeansCategoryFeature文件..')
        kmeans = StandardVersion.loadModel('KmeansNullFeature%d.model' % n_components)
    else:
        print('开始对类别特征训练Kmeans模型...')
        kmeans = KMeans(n_clusters=n_components).fit(df_train_null)

        print('训练完毕')
        StandardVersion.saveModel(kmeans, 'KmeansNullFeature%d.model' % n_components)

    labels_train = kmeans.predict(df_train_null)
    labels_test = kmeans.predict(df_test_null)
    df_train['KmeansNull'] = labels_train.tolist()
    df_test['KmeansNull'] = labels_test.tolist()

    print(df_train['KmeansNull'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmeansNull'], 'KmeansNullFeature')
    return df_train, df_test, column_name



# 笛卡尔分群作为新特征 8列
# example:descartesGroupNewFeature(df_train,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def descartesGroupNewFeature(data_origin, n1, n2, n3, s1, s2, s3):
    print('in descartesGroupNewFeature..')
    column_name = []
    for i in range(8):
        column_name.append('seg%d' % (i + 1))

    data = data_origin.copy(deep=True)

    data['seg1'] = 0;
    data['seg2'] = 0;
    data['seg3'] = 0;
    data['seg4'] = 0
    data['seg5'] = 0;
    data['seg6'] = 0;
    data['seg7'] = 0;
    data['seg8'] = 0

    data.loc[(data[n1] < s1) & (data[n2] < s2) & (data[n3] < s3), 'seg1'] = 1
    data.loc[(data[n1] < s1) & (data[n2] < s2) & (data[n3] >= s3), 'seg2'] = 1
    data.loc[(data[n1] < s1) & (data[n2] >= s2) & (data[n3] < s3), 'seg3'] = 1
    data.loc[(data[n1] < s1) & (data[n2] >= s2) & (data[n3] >= s3), 'seg4'] = 1
    data.loc[(data[n1] >= s1) & (data[n2] < s2) & (data[n3] < s3), 'seg5'] = 1
    data.loc[(data[n1] >= s1) & (data[n2] < s2) & (data[n3] >= s3), 'seg6'] = 1
    data.loc[(data[n1] >= s1) & (data[n2] >= s2) & (data[n3] < s3), 'seg7'] = 1
    data.loc[(data[n1] >= s1) & (data[n2] >= s2) & (data[n3] >= s3), 'seg8'] = 1

    return data, column_name


# 缺失值个数
def isNullCount(df_train, df_test):
    print('in isNullCount')
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()
    return df_train, df_test


def nullCountcut(df_train, df_test):
    import StandardVersion
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()

    x1_d, x1_iv, x1_cut, x1_woe = mono_bin(df_train.bad, df_train.nunNum)
    df_train['null_count'] = fenxiang(df_train, 'nunNum', x1_cut)
    df_test['null_count'] = fenxiang(df_test, 'nunNum', x1_cut)

    # 转为one-hot编码 4列
    # df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['null_count'], 'null_seg')
    # return df_train, df_test, column_name

    return df_train, df_test

def mono_bin(Y, X, n=4):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n,duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/good)/((1-d3['rate'])/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua=X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    return d4,iv,cut,woe

def fenxiang(data, featureName, cut):
    newFeatureName = featureName + '_cut'
    score_list = data[featureName]
    bins = cut # 分箱的区间
    score_cat = pd.cut(score_list,bins) # 分箱
    print(cut)
    l = len(cut) - 1
    print(l)
    data[newFeatureName] =pd.cut(data[featureName],bins,labels = range(0,l))
    print(data[newFeatureName].head())
    return data[newFeatureName]