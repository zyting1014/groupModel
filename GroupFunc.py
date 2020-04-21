"""
    分群作为新特征函数 被StandardVersion类调用
"""
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 决策树划分1
def decisionTreeMethod1(data_origin):
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
    data = data_origin.copy(deep=True)
    featureList = ['nasrdw_recd_date','var_jb_23','creditlimitamount_4']
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


# 高斯混合模型 用所有变量聚类
def getGMMCategoryFeature(df_train, df_test, feature_categorical, n_components=4):
    import StandardVersion
    print(feature_categorical)

    x_train = df_train[feature_categorical].copy().fillna(-99999)
    x_test = df_test[feature_categorical].copy().fillna(-99999)
    x_train = StandardVersion.proprocessCateory(x_train, feature_categorical)
    x_test = StandardVersion.proprocessCateory(x_test, feature_categorical)
    import os
    if os.path.exists('model/GMMCategoryFeature.model'):
        print('加载GMMCategoryFeature文件..')
        gmm = StandardVersion.loadModel('GMMCategoryFeature')
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(x_train)
        print('训练完毕')
        StandardVersion.saveModel(gmm,'model/GMMCategoryFeature.model')

    labels_train = gmm.predict(x_train)
    labels_test = gmm.predict(x_test)
    df_train['gmm'] = labels_train.tolist()
    df_test['gmm'] = labels_test.tolist()

    # 转为one-hot编码 4列
    df_train, df_test = StandardVersion.cateToOneHot(df_train, df_test, ['gmm'])
    return df_train, df_test



# 笛卡尔分群作为新特征 8列
# example:descartesGroupNewFeature(df_train,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def descartesGroupNewFeature(data_origin, n1, n2, n3, s1, s2, s3):
    data = data_origin.copy(deep=True)

    data['seg1'] = 0; data['seg2'] = 0; data['seg3'] = 0; data['seg4'] = 0
    data['seg5'] = 0; data['seg6'] = 0; data['seg7'] = 0; data['seg8'] = 0

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
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()
    return df_train, df_test
