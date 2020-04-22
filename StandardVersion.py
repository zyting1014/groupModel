# coding: utf-8
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import ParseData, Evaluation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.externals import joblib
import GroupFunc
import os

"""
    未进行分群 一个模型
"""

# 将参数写成字典下形式
param = {'num_leaves': 150, 'objective': 'binary', 'max_depth': 7, 'learning_rate': .05, 'max_bin': 200,
                 'metric': ['auc', 'binary_logloss']}
# param = {'objective': 'binary', 'learning_rate': .05,'metric': ['auc', 'binary_logloss']}


# 将类别特征转为str 这样不用one-hot就可以直接训练 但由于特征稀疏效果不好 故目前不加入30+维特征
def proprocessCateory(data, feature_categorical):
    lbl = preprocessing.LabelEncoder()
    for feature in feature_categorical:
        data[feature] = lbl.fit_transform(data[feature].astype(str))
        data[feature] = data[feature].astype('category')
    return data


def cateToOneHot(df_train, df_test, featureList,prefixName = ''):
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

        columnName = []
        for i in range(trainNewColumn.shape[1]):
            columnName.append(prefixName + str(i))
    df_train_new = pd.concat([df_train_new, pd.DataFrame(trainNewColumn,columns=columnName)], axis=1)
    df_test_new = pd.concat([df_test_new, pd.DataFrame(testNewColumn,columns=columnName)], axis=1)
    df_train = pd.concat([df_train, df_train_new], axis=1)
    df_test = pd.concat([df_test, df_test_new], axis=1)
    print('转化后特征数量为%s' % (df_train.shape[1]))
    return df_train, df_test


def getFeatureCategorical(data):
    import pandas.api.types as types
    feature_categorical = []
    for column in list(data.columns):
        if types.is_object_dtype(data[column]):
            feature_categorical.append(column)
    return feature_categorical


def getTrainTestSample(df_train, df_test, feature_categorical):
    target = 'bad'
    # feature_categorical.append(target)

    X_train = df_train.drop(feature_categorical, axis=1).drop(target, axis=1)
    X_test = df_test.drop(feature_categorical, axis=1).drop(target, axis=1)
    y_train = df_train[target]
    y_test = df_test[target]


    return X_train, y_train, X_test, y_test


def trainModel(X_train, y_train, X_test, y_test, round=1000):
    # 创建成lgb特征的数据集格式
    # print('训练数据维度：%s'%(X_train.shape))
    # print('测试数据维度：%s' % (X_test.shape))
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


def saveModel(model,modelName):
    path = os.path.join('model',modelName)
    joblib.dump(model, path)


def loadModel(modelName):
    path = os.path.join('model', modelName)
    return joblib.load(path)


# 集成所有分群生成新特征的函数
def getNewFeature(df_train, df_test, feature_categorical):

    # 决策树分群1
    # df_train = GroupFunc.decisionTreeMethod1(df_train)
    # df_test = GroupFunc.decisionTreeMethod1(df_test)
    # 决策树分群2
    # df_train = GroupFunc.decisionTreeMethod2(df_train)
    # df_test = GroupFunc.decisionTreeMethod2(df_test)
    # 空值特征数
    # df_train, df_test = GroupFunc.isNullCount(df_train, df_test)
    # 空/非空特征lda+GMM聚类
    # df_train, df_test = GroupFunc.getGMMNullFeature(df_train, df_test)
    # 类别特征GMM聚类
    df_train, df_test = GroupFunc.getGMMCategoryFeature(df_train, df_test, feature_categorical, 4)

    return df_train, df_test


def main():
    # df_train, df_test = ParseData.loadPartData()
    df_train, df_test = ParseData.loadData()
    feature_categorical = getFeatureCategorical(df_train)
    # df_train, df_test = getNewFeature(df_train, df_test, feature_categorical)
    x_train, y_train, x_test, y_test = getTrainTestSample(df_train, df_test, feature_categorical)
    gbm, y_pred = trainModel(x_train, y_train, x_test, y_test)
    Evaluation.getKsValue(y_test, y_pred)
    featureImportance(gbm)


if __name__ == '__main__':
    main()
