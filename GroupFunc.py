"""
    分群作为新特征函数 被StandardVersion类调用
"""
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os


# 决策树划分1
def decisionTreeMethod1(data_origin):
    print('in decisionTreeMethod1..')
    data = data_origin.copy(deep=True)
    featureList = ['var_jb_23', 'var_jb_28', 'nasrdw_recd_date']
    data[featureList].fillna(-99999)

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
        (data['var_jb_28'] <= 4.5) & (data['var_jb_23'] > 27.5) & (data['nasrdw_recd_date'] <= 20181023), 'seg4'] = 1
    data_origin.loc[(data['var_jb_28'] > 4.5) & (data['nasrdw_recd_date'] <= 20181011), 'seg5'] = 1
    data_origin.loc[(data['var_jb_28'] > 4.5) & (data['nasrdw_recd_date'] > 20181011), 'seg6'] = 1

    return data_origin


# 决策树划分2
def decisionTreeMethod2(data_origin):
    print('in decisionTreeMethod2..')
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

    return data_origin


def decisionTreeMethod3(data, n1, s1, c1, cs1, c2, cs2):
    print('in decisionTreeMethod3..')

    data['seg1'] = 0
    data['seg2'] = 0
    data['seg3'] = 0
    data['seg4'] = 0

    data.loc[((data[n1] < s1) | (data[n1].isnull())) & ((data[c1] < cs1) | (data[c1].isnull())),'seg1'] = 1
    data.loc[((data[n1] < s1) | (data[n1].isnull())) & (data[c1] >= cs1),'seg2'] = 1
    data.loc[(data[n1] >= s1) & ((data[c2] < cs2) | (data[c2].isnull())), 'seg3'] = 1
    data.loc[(data[n1] >= s1) & (data[c2] >= cs2), 'seg4'] = 1

    return data


# 高斯混合模型 用所有变量聚类
def getGMMCategoryFeature(df_train, df_test, feature_categorical, n_components=4):
    print('in getGMMCategoryFeature..')
    import StandardVersion
    print(feature_categorical)

    x_train = df_train[feature_categorical].copy().fillna(-99999)
    x_test = df_test[feature_categorical].copy().fillna(-99999)
    x_train = StandardVersion.proprocessCateory(x_train, feature_categorical)
    x_test = StandardVersion.proprocessCateory(x_test, feature_categorical)

    if os.path.exists('GMMCategoryFeature%d.model' % n_components):
        print('加载GMMCategoryFeature文件..')
        gmm = StandardVersion.loadModel('GMMCategoryFeature%d.model' % n_components)
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(x_train)
        print('训练完毕')
        StandardVersion.saveModel(gmm, 'GMMCategoryFeature%d.model' % n_components)

    labels_train = gmm.predict(x_train)
    labels_test = gmm.predict(x_test)
    df_train['gmm'] = labels_train.tolist()
    df_test['gmm'] = labels_test.tolist()

    # 转为one-hot编码
    df_train, df_test = StandardVersion.cateToOneHot(df_train, df_test, ['gmm'], 'GMMCategoryFeature')
    # df_train = df_train.drop('gmm', axis=1)
    # df_test = df_test.drop('gmm', axis=1)
    return df_train, df_test


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


    if os.path.exists('model/GMMNullFeature.model'):
        print('加载GMMCategoryFeature文件..')
        gmm = StandardVersion.loadModel('GMMNullFeature.model')
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(df_train_null)

        print('训练完毕')
        StandardVersion.saveModel(gmm, 'GMMNullFeature.model')

    labels_train = gmm.predict(df_train_null)
    labels_test = gmm.predict(df_test_null)
    df_train['gmmNull'] = labels_train.tolist()
    df_test['gmmNull'] = labels_test.tolist()

    print(df_train['gmmNull'].head())

    # 转为one-hot编码 4列
    df_train, df_test = StandardVersion.cateToOneHot(df_train, df_test, ['gmmNull'], 'GMMNullFeature')
    return df_train, df_test


# 笛卡尔分群作为新特征 8列
# example:descartesGroupNewFeature(df_train,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def descartesGroupNewFeature(data_origin, n1, n2, n3, s1, s2, s3):
    print('in descartesGroupNewFeature..')

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
    return data


# 缺失值个数
def isNullCount(df_train, df_test):
    print('in isNullCount')
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()
    return df_train, df_test








