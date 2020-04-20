# coding: utf-8
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import ParseData, Evaluation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

"""
    未进行分群 一个模型
"""

# 将参数写成字典下形式
param = {'num_leaves': 150, 'objective': 'binary', 'max_depth': 7, 'learning_rate': .05, 'max_bin': 200,
                 'metric': ['auc', 'binary_logloss']}

def proprocessCateory(data, feature_categorical):
    lbl = preprocessing.LabelEncoder()
    for feature in feature_categorical:
        data[feature] = lbl.fit_transform(data[feature].astype(str))
    return data

def cateToOneHot(df_train, df_test, featureList):
    print('将%s转化为one-hot编码，转化前特征数量为%d' % (featureList, df_train.shape[1]))
    enc = OneHotEncoder()
    df_train_new = pd.DataFrame()
    df_test_new = pd.DataFrame()
    for feature in featureList:
        trainFeatureArray = np.array([df_train[feature]]).T
        testFeatureArray = np.array([df_test[feature]]).T
        featureArray = np.vstack((trainFeatureArray, testFeatureArray))
        enc.fit(featureArray)
        trainNewColumn = enc.transform(trainFeatureArray).toarray()
        testNewColumn = enc.transform(testFeatureArray).toarray()

        df_train_new = pd.concat([df_train_new, pd.DataFrame(trainNewColumn)], axis=1)
        df_test_new = pd.concat([df_test_new, pd.DataFrame(testNewColumn)], axis=1)
    df_train = pd.concat([df_train, df_train_new], axis=1)
    df_test = pd.concat([df_test, df_test_new], axis=1)
    print('转化后特征数量为%s' % (df_train.shape[1]))
    return df_train, df_test

# 获得类别特征
def getFeatureCategorical(data):
    import pandas.api.types as types
    feature_categorical = []
    for column in list(data.columns):
        if types.is_object_dtype(data[column]):
            feature_categorical.append(column)
    return feature_categorical


def getTrainTestSample(df_train, df_test, feature_categorical):
    target = 'bad'
    feature_categorical.append(target)
    # y_train = pd.get_dummies(df_train, columns=feature_categorical)
    # X_train = pd.get_dummies(df_test, columns=feature_categorical)
    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train.drop(feature_categorical, axis=1)
    X_test = df_test.drop(feature_categorical, axis=1)
    # X_train = df_train
    # X_test = df_test
    return X_train, y_train, X_test, y_test


def trainModel(X_train, y_train, X_test, y_test, round=1000):
    # 创建成lgb特征的数据集格式
    # print('训练数据维度：%s'%(X_train.shape))
    # print('测试数据维度：%s' % (X_test.shape))
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print('Start training...')
    gbm = lgb.train(param, lgb_train, num_boost_round=round, valid_sets=lgb_eval, early_stopping_rounds=50)
    # print('Save model...')
    # gbm.save_model('model.txt')
    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The auc score is:', roc_auc_score(y_test, y_pred))
    return gbm, y_pred


def featureImportance(gbm):
    lgb.plot_importance(gbm, max_num_features=10)
    plt.show()

    importance = gbm.feature_importance(importance_type='split')
    feature_name = gbm.feature_name()
    # for (feature_name,importance) in zip(feature_name,importance):
    #     print (feature_name,importance)
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': list(importance)})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    print(feature_importance.head(10))
    feature_importance.to_csv('feature_importance.csv', index=False)


def main():
    df_train, df_test = ParseData.loadPartData()
    # df_train, df_test = ParseData.loadData()

    feature_categorical = getFeatureCategorical(df_train)
    df_train = proprocessCateory(df_train, feature_categorical)
    df_test = proprocessCateory(df_test, feature_categorical)
    # onehotList = ['var_jb_2','var_jb_3','var_jb_16','var_jb_17']
    onehotList = ['var_jb_2']
    df_train, df_test = cateToOneHot(df_train, df_test, onehotList)

    X_train, y_train, X_test, y_test = getTrainTestSample(df_train, df_test, feature_categorical)
    gbm, y_pred = trainModel(X_train, y_train, X_test, y_test)
    Evaluation.getKsValue(y_test, y_pred)
    featureImportance(gbm)


if __name__ == '__main__':
    main()
