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
import os, sys
import Tools
import ParseData
import kmedoids


# 决策树划分1 特定策略实现（决策树）
def decisionTreeMethod1(data_origin, is_segmentation=True):
    print('in %s' % sys._getframe().f_code.co_name)
    column_name = []
    for i in range(6):
        column_name.append('seg%d' % (i + 1))
    data = data_origin.copy(deep=True)
    feature_list = ['var_jb_23', 'var_jb_28', 'nasrdw_recd_date']
    if is_segmentation:
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


# 决策树划分2 特定策略实现（决策树）
def decisionTreeMethod2(data_origin, is_segmentation=True):
    print('in %s' % sys._getframe().f_code.co_name)
    column_name = []
    for i in range(4):
        column_name.append('seg%d' % (i + 1))
    data = data_origin.copy(deep=True)
    feature_list = ['nasrdw_recd_date', 'var_jb_23', 'creditlimitamount_4']
    if is_segmentation:
        data[feature_list] = data[feature_list].fillna(-99999)

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


# 决策树划分 分为4个群
def decisionTreeMethod3(data, n1, s1, c1, cs1, c2, cs2, is_segmentation=True):
    print('in %s' % sys._getframe().f_code.co_name)
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


# 决策树划分 分为4个群 12个新变量
def decisionTreeMethod4(data, n1, s1, c1, cs1, c2, cs2, cc1, ccs1, cc2, ccs2, cc3, ccs3, cc4, ccs4, is_segmentation=True):
    print('in %s' % sys._getframe().f_code.co_name)
    column_name = []
    for i in range(4):
        column_name.append('seg%d' % (i + 1))

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


    data.loc[((data[n1] < s1) | (data[n1].isnull())) & ((data[c1] < cs1) | (data[c1].isnull())), 'seg1'] = 1
    data.loc[((data[n1] < s1) | (data[n1].isnull())) & (data[c1] >= cs1), 'seg2'] = 1
    data.loc[(data[n1] >= s1) & ((data[c2] < cs2) | (data[c2].isnull())), 'seg3'] = 1
    data.loc[(data[n1] >= s1) & (data[c2] >= cs2), 'seg4'] = 1

    data.loc[((data['seg1'] == 1) & ((data[cc1] < ccs1) | data[cc1].isnull())), 'seg5'] = 1
    data.loc[((data['seg1'] == 1) & (data[cc1] >= ccs1)), 'seg6'] = 1
    data.loc[((data['seg2'] == 1) & ((data[cc2] < ccs2) | data[cc2].isnull())), 'seg7'] = 1
    data.loc[((data['seg2'] == 1) & (data[cc2] >= ccs2)), 'seg8'] = 1
    data.loc[((data['seg3'] == 1) & ((data[cc3] < ccs3) | data[cc3].isnull())), 'seg9'] = 1
    data.loc[((data['seg3'] == 1) & (data[cc3] < ccs3)), 'seg10'] = 1
    data.loc[((data['seg4'] == 1) & ((data[cc4] < ccs4) | data[cc4].isnull())), 'seg11'] = 1
    data.loc[((data['seg4'] == 1) & (data[cc4] >= ccs4)), 'seg12'] = 1

    return data, column_name

# 决策树划分2 特定策略实现(决策树) 新 除日期外重要特征
def decisionTreeMethod1New(data_origin,is_segmentation=True):
    print('in %s' % sys._getframe().f_code.co_name)
    column_name = []
    for i in range(3):
        column_name.append('seg%d' % (i + 1))
    data = data_origin.copy(deep=True)
    feature_list = ['creditlimitamount_4']
    if is_segmentation:
        data = data[feature_list].fillna(-99999)

    data['seg1'] = 0
    data['seg2'] = 0
    data['seg3'] = 0


    data_origin.loc[
        (data['creditlimitamount_4'] <= 299.5), 'seg1'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 299.5) & (data['creditlimitamount_4'] <= 65954.0), 'seg2'] = 1
    data_origin.loc[
        (data['creditlimitamount_4'] > 65954.0), 'seg3'] = 1

    return data_origin, column_name


# 高斯混合模型 用所有变量/类别变量聚类
def getGMMCategoryFeature(df_train, df_test, n_components=4):
    print('in %s' % sys._getframe().f_code.co_name)
    import StandardVersion
    feature_categorical = Tools.feature_categorical
    print(feature_categorical)

    gmm_list = feature_categorical + Tools.iv_more_than_point_one
    print('特征数量：%d' % (len(gmm_list)))

    x_train = df_train[gmm_list].copy().fillna(-99999)
    x_test = df_test[gmm_list].copy().fillna(-99999)
    x_train = StandardVersion.proprocessCateory(x_train, feature_categorical)
    x_test = StandardVersion.proprocessCateory(x_test, feature_categorical)

    feature_num = x_train.shape[1]

    if ParseData.existModel('GMMCategoryFeature%d_%d.model' % (n_components, feature_num)):
        print('加载GMMCategoryFeature文件..')
        gmm = ParseData.loadModel('GMMCategoryFeature%d_%d.model' % (n_components, feature_num))
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components, reg_covar=0.0001).fit(x_train)  # 可以调reg_covar=0.0001
        print('训练完毕')
        ParseData.saveModel(gmm, 'GMMCategoryFeature%d_%d.model' % (n_components, feature_num))

    labels_train = gmm.predict(x_train)
    labels_test = gmm.predict(x_test)

    df_train['gmm'] = labels_train.tolist()
    df_test['gmm'] = labels_test.tolist()

    # 转为one-hot编码
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['gmm'], 'GMMCategoryFeature')
    # df_train = df_train.drop('gmm', axis=1)
    # df_test = df_test.drop('gmm', axis=1)
    return df_train, df_test, column_name


# 高斯混合模型 用空/非空变量进行pca降维 再聚类
def getGMMNullFeature(df_train, df_test, n_components=4):
    print('in %s' % sys._getframe().f_code.co_name)
    import StandardVersion
    X_train = df_train.copy()
    X_test = df_test.copy()

    # 空值为1 非空为0 降维
    print('正在将缺失值设为0 非缺失值设为1..')
    df_train_null = X_train.where(X_train.isnull(), 0).fillna(1).astype(int)
    df_test_null = X_test.where(X_test.isnull(), 0).fillna(1).astype(int)

    # 加载pca模型

    pca_component = 10
    if ParseData.existModel('PCANullFeature%d.model' % pca_component):
        print('加载PCANullFeature文件..')
        pca = ParseData.loadModel('PCANullFeature%d.model' % pca_component)
    else:
        pca = PCA(n_components=pca_component)
        pca.fit(df_train_null.values)
        ParseData.saveModel(pca, 'PCANullFeature%d.model' % pca_component)

    df_train_null = pca.transform(df_train_null.values)
    df_test_null = pca.transform(df_test_null.values)
    print(df_train_null.shape)
    print(df_test_null.shape)

    # 加载GMM空-非空模型
    if ParseData.existModel('GMMNullFeature%d.model' % n_components):
        print('加载GMMNullFeature文件..')
        gmm = ParseData.loadModel('GMMNullFeature%d.model' % n_components)
    else:
        print('开始对类别特征训练GMM模型...')
        gmm = GMM(n_components=n_components).fit(df_train_null)

        print('训练完毕')
        ParseData.saveModel(gmm, 'GMMNullFeature%d.model' % n_components)

    labels_train = gmm.predict(df_train_null)
    labels_test = gmm.predict(df_test_null)
    df_train['gmmNull'] = labels_train.tolist()
    df_test['gmmNull'] = labels_test.tolist()

    print(df_train['gmmNull'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['gmmNull'], 'GMMNullFeature')
    return df_train, df_test, column_name


# k-means 用空/非空变量进行pca降维 再聚类
def getKmeansNullFeature(df_train, df_test, n_components=4):
    print('in %s' % sys._getframe().f_code.co_name)
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
    print(df_test_null)

    if ParseData.existModel('KmeansNullFeature%d.model' % n_components):
        print('加载KmeansCategoryFeature文件..')
        kmeans = ParseData.loadModel('KmeansNullFeature%d.model' % n_components)
    else:
        print('开始对类别特征训练Kmeans模型...')
        kmeans = KMeans(n_clusters=n_components).fit(df_train_null)

        print('训练完毕')
        ParseData.saveModel(kmeans, 'KmeansNullFeature%d.model' % n_components)

    labels_train = kmeans.predict(df_train_null)
    labels_test = kmeans.predict(df_test_null)
    df_train['KmeansNull'] = labels_train.tolist()
    df_test['KmeansNull'] = labels_test.tolist()

    print(df_train['KmeansNull'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmeansNull'],
                                                                  'KmeansNullFeature')
    return df_train, df_test, column_name


# k-means 所有变量 平滑处理后 直接聚类
def getKmeansAllFeature(df_train, df_test, n_components=4):
    import StandardVersion
    print('in %s' % sys._getframe().f_code.co_name)

    feature_categorical = Tools.feature_categorical
    df_train_smooth = df_train.copy()
    df_test_smooth = df_test.copy()

    # 获得要提取的特征列
    iv_more_than_point_one = Tools.iv_more_than_point_one
    feature_categorical = Tools.feature_categorical
    if ParseData.TYPE == 'OOT_noDate':
        iv_more_than_point_one = list(set(iv_more_than_point_one) - set(Tools.feature_date))
        feature_categorical = list(set(feature_categorical) - set(Tools.feature_date))

    kmeans_list = iv_more_than_point_one + feature_categorical
    df_train_smooth = df_train_smooth[kmeans_list]
    df_test_smooth = df_test_smooth[kmeans_list]

    # 缺失值处理 类别特征str化 平滑处理
    # for column in Tools.iv_more_than_point_one:
    #     df_train_smooth[column] = df_train_smooth[column].fillna(df_train_smooth[column].mean())
    #     df_test_smooth[column] = df_test_smooth[column].fillna(df_test_smooth[column].mean())


    df_train_smooth = df_train_smooth.fillna(-99999)
    df_test_smooth = df_test_smooth.fillna(-99999)
    df_train_smooth = StandardVersion.proprocessCateory(df_train_smooth, feature_categorical)
    df_test_smooth = StandardVersion.proprocessCateory(df_test_smooth, feature_categorical)
    df_train_smooth = Tools.apply_log1p_transformation(df_train_smooth, iv_more_than_point_one)
    df_test_smooth = Tools.apply_log1p_transformation(df_test_smooth, iv_more_than_point_one)


    # 开始kmeans训练
    if ParseData.existModel('KmeansAllFeature%d.model' % n_components):
        print('加载KmeansAllFeature文件..')
        kmeans = ParseData.loadModel('KmeansAllFeature%d.model' % n_components)
    else:
        print('开始训练kmeans模型..')
        kmeans = KMeans(n_clusters=n_components).fit(df_train_smooth)
        print('训练完毕')
        ParseData.saveModel(kmeans, 'KmeansAllFeature%d.model' % n_components)

    labels_train = kmeans.predict(df_train_smooth)
    labels_test = kmeans.predict(df_test_smooth)
    df_train['KmeansAll'] = labels_train.tolist()
    df_test['KmeansAll'] = labels_test.tolist()

    print(df_train['KmeansAll'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmeansAll'],
                                                                  'KmeansAllFeature')

    return df_train, df_test, column_name


# 先用pca降维 再进行getKmeansAllFeature方法
def getKmeansAllFeaturePCA(df_train, df_test, n_components=4):
    import StandardVersion
    print('in %s' % sys._getframe().f_code.co_name)

    feature_categorical = Tools.feature_categorical
    df_train_smooth = df_train.copy()
    df_test_smooth = df_test.copy()

    # 获得要提取的特征列
    iv_more_than_point_one = Tools.iv_more_than_point_one
    feature_categorical = Tools.feature_categorical
    if ParseData.TYPE == 'OOT_noDate':
        iv_more_than_point_one = list(set(iv_more_than_point_one) - set(Tools.feature_date))
        feature_categorical = list(set(feature_categorical) - set(Tools.feature_date))

    kmeans_list = iv_more_than_point_one + feature_categorical

    df_train_smooth = df_train_smooth[kmeans_list]
    df_test_smooth = df_test_smooth[kmeans_list]

    # 缺失值处理 类别特征str化 平滑处理
    df_train_smooth = df_train_smooth.fillna(-99999)
    df_test_smooth = df_test_smooth.fillna(-99999)
    df_train_smooth = StandardVersion.proprocessCateory(df_train_smooth, feature_categorical)
    df_test_smooth = StandardVersion.proprocessCateory(df_test_smooth, feature_categorical)
    df_train_smooth = Tools.apply_log1p_transformation(df_train_smooth, iv_more_than_point_one)
    df_test_smooth = Tools.apply_log1p_transformation(df_test_smooth, iv_more_than_point_one)

    # PCA降维
    # 开始pca训练
    if ParseData.existModel('PCAAllFeature.model'):
        print('PCAAllFeature..')
        pca = ParseData.loadModel('PCAAllFeature.model')
    else:
        pca = PCA(0.95)
        pca.fit(df_train_smooth.values)
        ParseData.saveModel(pca, 'PCAAllFeature.model')

    df_train_smooth = pca.transform(df_train_smooth)
    df_test_smooth = pca.transform(df_test_smooth)


    # 开始kmeans训练
    if ParseData.existModel('KmeansAllFeaturePCA%d.model' % n_components):
        print('加载KmeansAllFeature文件..')
        kmeans = ParseData.loadModel('KmeansAllFeaturePCA%d.model' % n_components)
    else:
        print('开始训练kmeans模型..')
        kmeans = KMeans(n_clusters=n_components).fit(df_train_smooth)
        print('训练完毕')
        ParseData.saveModel(kmeans, 'KmeansAllFeaturePCA%d.model' % n_components)

    labels_train = kmeans.predict(df_train_smooth)
    labels_test = kmeans.predict(df_test_smooth)
    df_train['KmeansAllPCA'] = labels_train.tolist()
    df_test['KmeansAllPCA'] = labels_test.tolist()

    print(df_train['KmeansAllPCA'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmeansAllPCA'],
                                                                  'KmeansAllFeaturePCA')

    return df_train, df_test, column_name


# k-means 所有变量 不经过iv值过滤 平滑处理后 直接聚类
def getKmeansAllFeatureNoFilter(df_train, df_test, n_components=4):
    import StandardVersion
    print('in %s' % sys._getframe().f_code.co_name)

    feature_categorical = Tools.feature_categorical
    df_train_smooth = df_train.copy()
    df_test_smooth = df_test.copy()

    # 获得要提取的特征列
    kmeans_list = Tools.not_feature_categorical + feature_categorical
    df_train_smooth = df_train_smooth[kmeans_list]
    df_test_smooth = df_test_smooth[kmeans_list]

    # 缺失值处理 类别特征str化 平滑处理
    df_train_smooth = df_train_smooth.fillna(-99999)
    df_test_smooth = df_test_smooth.fillna(-99999)
    df_train_smooth = StandardVersion.proprocessCateory(df_train_smooth, feature_categorical)
    df_test_smooth = StandardVersion.proprocessCateory(df_test_smooth, feature_categorical)
    df_train_smooth = Tools.apply_log1p_transformation(df_train_smooth, Tools.iv_more_than_point_one)
    df_test_smooth = Tools.apply_log1p_transformation(df_test_smooth, Tools.iv_more_than_point_one)


    # 开始kmeans训练
    if ParseData.existModel('KmeansAllFeatureNoFilter%d.model' % n_components):
        print('加载KmeansAllFeature文件..')
        kmeans = ParseData.loadModel('KmeansAllFeatureNoFilter%d.model' % n_components)
    else:
        print('开始训练kmeans模型..')
        kmeans = KMeans(n_clusters=n_components).fit(df_train_smooth)
        print('训练完毕')
        ParseData.saveModel(kmeans, 'KmeansAllFeatureNoFilter%d.model' % n_components)

    labels_train = kmeans.predict(df_train_smooth)
    labels_test = kmeans.predict(df_test_smooth)
    df_train['KmeansAllNoFilter'] = labels_train.tolist()
    df_test['KmeansAllNoFilter'] = labels_test.tolist()

    print(df_train['KmeansAllNoFilter'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmeansAllNoFilter'],
                                                                  'KmeansAllFeatureNoFilter')

    return df_train, df_test, column_name


# k-means 所有变量 平滑处理后 直接聚类
def getKmediodAllFeature(df_train, df_test, n_components=4):
    import StandardVersion
    print('in %s' % sys._getframe().f_code.co_name)

    feature_categorical = Tools.feature_categorical
    df_train_smooth = df_train.copy()
    df_test_smooth = df_test.copy()

    # 获得要提取的特征列
    kmediod_list = Tools.iv_more_than_point_one + feature_categorical
    df_train_smooth = df_train_smooth[kmediod_list]
    df_test_smooth = df_test_smooth[kmediod_list]


    df_train_smooth = df_train_smooth.fillna(-99999)
    df_test_smooth = df_test_smooth.fillna(-99999)
    df_train_smooth = StandardVersion.proprocessCateory(df_train_smooth, feature_categorical)
    df_test_smooth = StandardVersion.proprocessCateory(df_test_smooth, feature_categorical)
    df_train_smooth = Tools.apply_log1p_transformation(df_train_smooth, Tools.iv_more_than_point_one)
    df_test_smooth = Tools.apply_log1p_transformation(df_test_smooth, Tools.iv_more_than_point_one)

    # 开始kmediod训练
    if ParseData.existModel('KmediodAllFeature%d.model' % n_components):
        print('加载KmediodAllFeature文件..')
        kmediod = ParseData.loadModel('KmediodAllFeature%d.model' % n_components)
    else:
        print('开始训练kmediod模型..')
        kmediod = kmedoids.KMediod(n_components).fit(df_train_smooth)
        print('训练完毕')
        ParseData.saveModel(kmediod, 'KmediodAllFeature%d.model' % n_components)

    labels_train = kmediod.predict(df_train_smooth)
    labels_test = kmediod.predict(df_test_smooth)
    df_train['KmediodAll'] = labels_train.tolist()
    df_test['KmediodAll'] = labels_test.tolist()

    print(df_train['KmediodAll'].head())

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['KmediodAll'],
                                                                  'KmediodAllFeature')

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


# 缺失值个数 作为新特征添加
def isNullCount(df_train, df_test):
    print('in isNullCount')
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()
    return df_train, df_test


# 按照缺失值个数分箱 分成多个群体
def nullCountcut(df_train, df_test):
    print('in %s' % sys._getframe().f_code.co_name)
    import StandardVersion
    df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    df_test['nunNum'] = df_test.isnull().sum(axis=1).tolist()

    x1_d, x1_iv, x1_cut, x1_woe = mono_bin(df_train.bad, df_train.nunNum)
    df_train['null_count'] = fenxiang(df_train, 'nunNum', x1_cut)
    df_test['null_count'] = fenxiang(df_test, 'nunNum', x1_cut)

    # 转为one-hot编码 4列
    df_train, df_test, column_name = StandardVersion.cateToOneHot(df_train, df_test, ['null_count'], 'null_seg')
    return df_train, df_test, column_name

# 单变量iv, cut, woe的计算
def mono_bin(Y, X, n=4):
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / good) / ((1 - d3['rate']) / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min')).reset_index(drop=True)
    woe = list(d4['woe'].round(3))
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    return d4, iv, cut, woe


# 变量分箱的实现
def fenxiang(data, feature_name, cut):
    new_feature_name = feature_name + '_cut'
    score_list = data[feature_name]
    bins = cut  # 分箱的区间
    score_cat = pd.cut(score_list, bins)  # 分箱
    print(cut)
    l = len(cut) - 1
    print(l)
    data[new_feature_name] = pd.cut(data[feature_name], bins, labels=range(0, l))
    print(data[new_feature_name].head())
    return data[new_feature_name]


# 移除日期特征
def removeDateColumn(data):
    import Tools
    origin_column = data.columns
    date_column = Tools.feature_date
    remove_column = set(origin_column).intersection(set(date_column))

    return data.drop(columns=remove_column, axis=1)
