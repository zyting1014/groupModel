# coding: utf-8
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import ParseData, Evaluation, EvaluateSegmentation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import GroupFunc
import os

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


def cateToOneHot(df_train, df_test, feature_list, prefix_name=''):
    print('将%s转化为one-hot编码，转化前特征数量为%d' % (feature_list, df_train.shape[1]))
    enc = OneHotEncoder()
    df_train_new = pd.DataFrame()
    df_test_new = pd.DataFrame()
    for feature in feature_list:
        train_feature_array = np.array([df_train[feature]]).T
        test_feature_array = np.array([df_test[feature]]).T
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


def getFeatureCategorical(data):
    import pandas.api.types as types
    import Tools
    feature_categorical = []
    for column in list(data.columns):
        if types.is_object_dtype(data[column]):
            feature_categorical.append(column)

    return feature_categorical


def getFeatureDate(data):
    import pandas.api.types as types
    feature_date = []
    for column in list(data.columns):
        # 字段名带date的 或以20开头的
        if str(column).find('date') != -1 or str(data[column][0]).startswith('20'):
            feature_date.append(column)
    return feature_date



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


# 集成所有分群生成新特征的函数
def getNewFeature(df_train, df_test, feature_categorical):
    # 决策树分群1
    # df_train, column_name = GroupFunc.decisionTreeMethod1(df_train, False)
    # df_test, column_name = GroupFunc.decisionTreeMethod1(df_test, False)
    # 决策树分群2
    # df_train, column_name = GroupFunc.decisionTreeMethod2(df_train, False)
    # df_test, column_name = GroupFunc.decisionTreeMethod2(df_test, False)
    # xgboost分群3
    # df_train, column_name = GroupFunc.decisionTreeMethod3(df_train, 'type_91|个人消费贷款', 0.5, 'var_jb_64', 13.5,
    #                                                       'var_jb_40', 0.5)
    # df_test, column_name = GroupFunc.decisionTreeMethod3(df_test, 'type_91|个人消费贷款', 0.5, 'var_jb_64', 13.5, 'var_jb_40',
    #                                                      0.5)
    # # xgboost分群4
    # df_train, column_name = GroupFunc.decisionTreeMethod3(df_train, 'creditlimitamount_4', 32188.5, 'var_jb_22', 13.5, 'nasrdw_recd_date', 20181024)
    # df_test, column_name = GroupFunc.decisionTreeMethod3(df_test, 'creditlimitamount_4', 32188.5, 'var_jb_22', 13.5, 'nasrdw_recd_date', 20181024)
    # 空值特征数
    # df_train, df_test = GroupFunc.isNullCount(df_train, df_test)
    # 空/非空特征lda+GMM聚类
    df_train, df_test, column_name = GroupFunc.getGMMNullFeature(df_train, df_test, 4)
    # 类别特征GMM聚类
    # df_train, df_test, column_name = GroupFunc.getGMMCategoryFeature(df_train, df_test, 4)

    # 空值特征数+分箱
    # df_train, df_test = GroupFunc.nullCountcut(df_train, df_test)

    # kmeans所有特征聚类
    # df_train, df_test, column_name = GroupFunc.getKmeansAllFeaturePCA(df_train, df_test, 2)

    # df_train, column_name = GroupFunc.decisionTreeMethod1New(df_train, False)
    # df_test, column_name = GroupFunc.decisionTreeMethod1New(df_test, False)

    # df_train, column_name = GroupFunc.decisionTreeMethod4(df_train, 'creditlimitamount_4', 32188.5, 'var_jb_94', 571.5,
    #                                                       'var_jb_22', 13.5,
    #                                                       'var_jb_15', 169, 'creditlimitamount_4', 5999.5,
    #                                                       'creditlimitamount_4', 245020, 'creditlimitamount_4', 80855)
    # df_test, column_name = GroupFunc.decisionTreeMethod4(df_test, 'creditlimitamount_4', 32188.5, 'var_jb_94', 571.5,
    #                                                      'var_jb_22', 13.5,
    #                                                      'var_jb_15', 169, 'creditlimitamount_4', 5999.5,
    #                                                      'creditlimitamount_4', 245020, 'creditlimitamount_4', 80855)

    # df_train, column_name = GroupFunc.decisionTreeMethod4(df_train, 'type_91|个人消费贷款', 0.5, 'var_jb_28', 4.5,
    #                                                       'var_jb_64', 13.5,
    #                                                       'mis_date_7', 20181010, 'latest5yearoverduebeginmonth_4', 2015.06494,
    #                                                       'var_jb_20', 0.0185000002, 'var_jb_40', 0.5)
    # df_test, column_name = GroupFunc.decisionTreeMethod4(df_test, 'type_91|个人消费贷款', 0.5, 'var_jb_28', 4.5,
    #                                                       'var_jb_64', 13.5,
    #                                                       'mis_date_7', 20181010, 'latest5yearoverduebeginmonth_4', 2015.06494,
    #                                                       'var_jb_20', 0.0185000002, 'var_jb_40', 0.5)


    return df_train, df_test


def main():
    # df_train, df_test = ParseData.loadPartData()
    # df_train, df_test = ParseData.loadData()
    df_train, df_test = ParseData.loadOOTData()
    # df_train, df_test = ParseData.loadOOT15Data()

    # df_train['nunNum'] = df_train.isnull().sum(axis=1).tolist()
    # df_train = df_train[df_train['nunNum'] < 150]
    # df_train = df_train.drop(columns=['nunNum'])

    feature_categorical = getFeatureCategorical(df_train)
    df_train, df_test = getNewFeature(df_train, df_test, feature_categorical)
    x_train, y_train, x_test, y_test = getTrainTestSample(df_train, df_test, feature_categorical)
    gbm, y_pred = trainModel(x_train, y_train, x_test, y_test)
    Evaluation.getKsValue(y_test, y_pred)
    featureImportance(gbm)


if __name__ == '__main__':
    main()
