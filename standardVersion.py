# coding: utf-8
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import parseData

# 将参数写成字典下形式
params = {'num_leaves': 150, 'objective': 'binary', 'max_depth': 7, 'learning_rate': .05, 'max_bin': 200,
          'metric': ['auc', 'binary_logloss']}


# 获得类别特征
def getFeatureCategorical(data):
    import pandas.api.types as types
    feature_categorical = []
    for column in list(data.columns):
        if types.is_object_dtype(data[column]):
            feature_categorical.append(column)
    return feature_categorical

def getTrainTestSample(df_train,df_test,feature_categorical):
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

def trainModel(X_train, y_train, X_test, y_test):
    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=50)
    print('Save model...')
    # gbm.save_model('model.txt')
    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The auc score is:', roc_auc_score(y_test, y_pred))


def main():
    df_train, df_test = parseData.loadData()
    feature_categorical = getFeatureCategorical(df_train)
    X_train, y_train, X_test, y_test = getTrainTestSample(df_train, df_test,feature_categorical)
    trainModel(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()