# coding: utf-8
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import ParseData, Evaluation, EvaluateSegmentation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import GroupFunc_old
import os, time
import pandas.api.types as types

"""
    未进行分群 一个模型
"""

# 将参数写成字典下形式
# param = {'num_leaves': 128, 'objective': 'binary', 'max_depth': 7, 'learning_rate': .05, 'max_bin': 200,
#                  'metric': ['auc', 'binary_logloss']}
param = {'objective': 'binary', 'learning_rate': .05, 'metric': ['auc', 'binary_logloss']}
print(param)


# 将类别特征转为str 这样不用one-hot就可以直接训练 但由于特征稀疏效果不好 故目前不加入30+维特征
def proprocessCateory(data, feature_categorical):
    lbl = preprocessing.LabelEncoder()
    for feature in feature_categorical:
        data[feature] = lbl.fit_transform(data[feature].astype(str))
        data[feature] = data[feature].astype('category')
    return data


# 将类别特征转为one-hot编码
def cateToOneHot(df_train, df_test, feature_list, prefix_name=''):
    print('将%s转化为one-hot编码，转化前特征数量为%d' % (feature_list, df_train.shape[1]))
    enc = OneHotEncoder()
    df_train_new = pd.DataFrame()
    df_test_new = pd.DataFrame()
    print(df_train[feature_list].isnull().sum())
    for feature in feature_list:
        train_feature_array = np.array([df_train[feature].astype(str)]).T
        test_feature_array = np.array([df_test[feature].astype(str)]).T
        feature_array = np.vstack((train_feature_array, test_feature_array))
        enc.fit(feature_array)
        train_new_column = enc.transform(train_feature_array).toarray()
        test_new_column = enc.transform(test_feature_array).toarray()

        column_name = []
        for i in range(train_new_column.shape[1]):
            column_name.append(prefix_name + str(i))
    df_train_new = pd.concat([df_train_new, pd.DataFrame(train_new_column, columns=column_name)], axis=1)
    df_test_new = pd.concat([df_test_new, pd.DataFrame(test_new_column, columns=column_name)], axis=1)
    df_train = pd.concat([df_train, df_train_new], axis=1)
    df_test = pd.concat([df_test, df_test_new], axis=1)
    print('转化后特征数量为%s' % (df_train.shape[1]))
    return df_train, df_test, column_name


# 获得类别特征
def getFeatureCategorical(data):
    import pandas.api.types as types
    import Tools
    feature_categorical = []
    for column in list(data.columns):
        if types.is_object_dtype(data[column]):
            feature_categorical.append(column)

    return feature_categorical


from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


# 将类别数小于100的类别特征转化为one-hot编码,再用pca降维
def CategoryPCA(df_train_, df_test_, feature_categorical):
    df_train = df_train_.copy()
    df_test = df_test_.copy()

    # 将cnt小于100的特征用PCA降维 添加到原始特征上
    feature_pca = []
    for feature in feature_categorical:
        if df_train[feature].nunique() <= 100:
            feature_pca.append(feature)

    df_train[feature_pca] = df_train[feature_pca].fillna('nan')
    df_test[feature_pca] = df_test[feature_pca].fillna('nan')

    df_train_new, df_test_new, feature_pca_new = cateToOneHot(df_train, df_test, feature_pca, 'cate_')

    print(df_train_new[feature_pca_new].head())

    pca = PCA(n_components=0.95)
    pca.fit(df_train_new[feature_pca_new].values)
    new_train = pca.transform(df_train_new[feature_pca_new].values)
    new_test = pca.transform(df_test_new[feature_pca_new].values)

    new_train = pd.DataFrame(new_train, columns=feature_pca_new[:new_train.shape[1]])
    new_test = pd.DataFrame(new_test, columns=feature_pca_new[:new_train.shape[1]])

    df_train = pd.concat([df_train_, new_train], axis=1)
    df_test = pd.concat([df_test_, new_test], axis=1)

    print(df_train.shape)
    print(df_test.shape)

    return df_train, df_test


# 获得日期特征
def getFeatureDate(data):
    feature_date = []
    for column in list(data.columns):
        # 字段名带date的 或以20开头的
        if str(column).find('date') != -1:
            feature_date.append(column)
            continue
        for j in range(200):
            if str(data[column][j]).startswith('201') and len(str(data[column][j])) > 10:
                feature_date.append(column)
                break
    return feature_date


# 将日期变量转为int
def parseDateToInt(data0, columns):
    def change(x):
        if isinstance(x, str):
            if x.find('/') != -1:
                y = time.strptime(x, '%Y/%m/%d %H:%M')
                x = str(y.tm_year)
                if len(str(y.tm_mon)) == 1:
                    x = x + '0'
                x = x + str(y.tm_mon)
                if len(str(y.tm_mday)) == 1:
                    x = x + '0'
                x = x + str(y.tm_mday)

            x = x.replace('.', '')
            x = x.replace(' ', '.')
            return x

    data = data0.copy(deep=True)
    for column in columns:
        if types.is_string_dtype(data[column]):
            if data[column].isnull().sum(axis=0) != 0:
                data[column] = data[column].apply(change)

                data[column] = pd.to_numeric(data[column], errors='coerce')
    #                 data[column] = data[column + '_new']
    return data


# 获得稀疏数据列 稀疏度阈值为0.2
def getNotSparseFeature(df_train):
    sparse_feature = []
    cnt = df_train.shape[0]
    for column in df_train.columns:
        null_rate = df_train[column].isnull().sum(axis=0) / cnt
        if null_rate > 0.2:
            sparse_feature.append(column)
    return sparse_feature


def getTrainTestSample(df_train, df_test, feature_categorical):
    target = 'bad'
    # feature_categorical.append(target)

    X_train = df_train.drop(feature_categorical, axis=1).drop(target, axis=1)
    X_test = df_test.drop(feature_categorical, axis=1).drop(target, axis=1)
    y_train = df_train[target]
    y_test = df_test[target]

    return X_train, y_train, X_test, y_test


# 使用原生lgb
def trainModel(X_train, y_train, X_test, y_test, round=2000):
    # 创建成lgb特征的数据集格式
    print(X_train.shape)
    print(X_test.shape)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print('Start training...')
    # print(categorical_feature)
    gbm = lgb.train(param, lgb_train, num_boost_round=round, valid_sets=lgb_eval,
                    early_stopping_rounds=50)
    # print('Save model...')
    # gbm.save_model('model.txt')
    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The auc score is:', roc_auc_score(y_test, y_pred))
    return gbm, y_pred


# 使用sklearn接口
def trainModelClassifier(X_train, y_train, X_test, y_test, round=1000):
    print('Start training...')
    gbm = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=round)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=50)

    # print('Save model...')
    # gbm.save_model('model.txt')
    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    print('The auc score is:', roc_auc_score(y_test, y_pred))
    return gbm, y_pred


def featureImportance(gbm):
    lgb.plot_importance(gbm, max_num_features=10)
    plt.show()

    importance = gbm.feature_importance(importance_type='split')
    feature_name = gbm.feature_name()

    for (name, value) in zip(feature_name, importance):
        print(name, value)
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': list(importance)})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    print(feature_importance)
    feature_importance.to_csv('feature_importance.csv', index=False)


def supervised_method(df_train, df_test):
    return df_train, df_test


def unsupervised_method(df_train, df_test):
    # 空值特征数
    # df_train, df_test = GroupFunc.isNullCount(df_train, df_test)
    # 空/非空特征lda+GMM聚类
    # df_train, df_test, column_name = GroupFunc.getGMMNullFeature(df_train, df_test, 4)
    # 类别特征GMM聚类
    # df_train, df_test, column_name = GroupFunc.getGMMCategoryFeature(df_train, df_test, 4)

    # 空值特征数+分箱
    # df_train, df_test = GroupFunc.nullCountcut(df_train, df_test)

    # kmeans所有特征聚类
    # df_train, df_test, column_name = GroupFunc.getKmeansAllFeaturePCA(df_train, df_test, 2)

    # df_train, column_name = GroupFunc.decisionTreeMethod1New(df_train, False)
    # df_test, column_name = GroupFunc.decisionTreeMethod1New(df_test, False)

    return df_train, df_test


# 集成所有分群生成新特征的函数
def getNewFeature(df_train, df_test):
    df_train, df_test = supervised_method(df_train, df_test)
    df_train, df_test = unsupervised_method(df_train, df_test)
    return df_train, df_test


def main():
    # df_train, df_test = ParseData.loadPartData()
    # df_train, df_test = ParseData.loadData()
    df_train, df_test = ParseData.loadOOTData()
    # df_train, df_test = ParseData.loadOOT15Data()

    # df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    # df_train = df_train[df_train['nunNum'] < 150]
    # df_train = df_train.drop(columns=['nunNum'])
    ##################################################
    # 日期特征处理
    feature_date = getFeatureDate(df_train)
    df_train = parseDateToInt(df_train, feature_date)
    df_test = parseDateToInt(df_test, feature_date)
    # 类别特征处理
    df_train, df_test = CategoryPCA(df_train, df_test, getFeatureCategorical(df_train))
    ##################################################



    feature_categorical = getFeatureCategorical(df_train)
    df_train, df_test = getNewFeature(df_train, df_test)


    x_train, y_train, x_test, y_test = getTrainTestSample(df_train, df_test, feature_categorical)
    gbm, y_pred = trainModel(x_train, y_train, x_test, y_test)
    Evaluation.getKsValue(y_test, y_pred)
    featureImportance(gbm)

    Evaluation.get_pos_neg_picture(y_test, y_pred)


if __name__ == '__main__':
    main()
