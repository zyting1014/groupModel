import numpy as np
from sklearn.metrics import roc_auc_score
import StandardVersion as baseline
import ParseData
import Evaluation

"""
    实现同时训练多个模型
"""
# 分群文件生成 生成3个训练、测试集
def descartesGroupDataToListOneHalf(df_train, df_test, n1, s1, c1, cs1):
    trainList = [
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & ((df_train[c1] < cs1) | (df_train[c1].isnull()))],
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & (df_train[c1] >= cs1)],
        df_train[(df_train[n1] >= s1)]
        ]
    testList = [
    df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & ((df_test[c1] < cs1) | (df_test[c1].isnull()))],
    df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & (df_test[c1] >= cs1)],
    df_test[(df_test[n1] >= s1)]
    ]
    return trainList, testList


# 分群文件生成 生成2个训练、测试集
def descartesGroupDataToListOne(df_train, df_test, n1, s1, onehot=False):
    df_train_fillnan = df_train.copy(deep=True)
    df_test_fillnan = df_test.copy(deep=True)
    df_train_fillnan[n1].fillna(-99999, inplace=True)
    df_test_fillnan[n1].fillna(-99999, inplace=True)
    if not onehot:
        trainList = [df_train[df_train_fillnan[n1] < s1], df_train[df_train_fillnan[n1] >= s1]]
        testList = [df_test[df_test_fillnan[n1] < s1], df_test[df_test_fillnan[n1] >= s1]]
    else:
        trainList = [df_train[df_train_fillnan[n1] == s1], df_train[df_train_fillnan[n1] != s1]]
        testList = [df_test[df_test_fillnan[n1] == s1], df_test[df_test_fillnan[n1] != s1]]

    return trainList, testList


def descartesGroupDataToListTwo(df_train, df_test, n1, s1, c1, cs1, c2, cs2):

    trainList = [df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & ((df_train[c1] < cs1) | (df_train[c1].isnull()))],
                 df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & (df_train[c1] >= cs1)],
                 df_train[(df_train[n1] >= s1) & ((df_train[c2] < cs2) | (df_train[c2].isnull()))],
                 df_train[(df_train[n1] >= s1) & (df_train[c2] >= cs2)],
                 ]
    testList = [
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & ((df_test[c1] < cs1) | (df_test[c1].isnull()))],
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & (df_test[c1] >= cs1)],
        df_test[(df_test[n1] >= s1) & ((df_test[c2] < cs2) | (df_test[c2].isnull()))],
        df_test[(df_test[n1] >= s1) & (df_test[c2] >= cs2)],
        ]


    checkTrainTestList(df_train, df_test, trainList, testList)

    return trainList, testList


# 分群文件生成 生成8个训练、测试集
# example:trainList,testList = descartesGroupDataToList
# (df_train,df_test,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def descartesGroupDataToListThree(df_train, df_test, n1, n2, n3, s1, s2, s3):
    df_train_fillnan = df_train.copy(deep=True)
    df_test_fillnan = df_test.copy(deep=True)
    df_train_fillnan[n1].fillna(-99999, inplace=True)
    df_train_fillnan[n2].fillna(-99999, inplace=True)
    df_train_fillnan[n3].fillna(-99999, inplace=True)
    df_test_fillnan[n1].fillna(-99999, inplace=True)
    df_test_fillnan[n2].fillna(-99999, inplace=True)
    df_test_fillnan[n3].fillna(-99999, inplace=True)
    trainList = []
    testList = []

    def abc(data, b1, b2, b3):
        if b1 == 0:
            data = data[data[n1] < s1]
        else:
            data = data[data[n1] >= s1]
        if b2 == 0:
            data = data[data[n2] < s2]
        else:
            data = data[data[n2] >= s2]
        if b3 == 0:
            data = data[data[n3] < s3]
        else:
            data = data[data[n3] >= s3]
        return data

    trainList = [abc(df_train_fillnan, 0, 0, 0), abc(df_train_fillnan, 0, 0, 1), abc(df_train_fillnan, 0, 1, 0),
                 abc(df_train_fillnan, 0, 1, 1),
                 abc(df_train_fillnan, 1, 0, 0), abc(df_train_fillnan, 1, 0, 1), abc(df_train_fillnan, 1, 1, 0),
                 abc(df_train_fillnan, 1, 1, 1)]
    testList = [abc(df_test_fillnan, 0, 0, 0), abc(df_test_fillnan, 0, 0, 1), abc(df_test_fillnan, 0, 1, 0),
                abc(df_test_fillnan, 0, 1, 1),
                abc(df_test_fillnan, 1, 0, 0), abc(df_test_fillnan, 1, 0, 1), abc(df_test_fillnan, 1, 1, 0),
                abc(df_test_fillnan, 1, 1, 1)]

    checkTrainTestList(df_train, df_test, trainList, testList)

    return trainList, testList


def checkTrainTestList(df_train, df_test, trainList, testList):
    trainCount = 0;
    testCount = 0
    print('分群数量为%d' % (len(trainList)))
    for i in range(len(trainList)):
        trainCount += trainList[i].shape[0]
        testCount += testList[i].shape[0]

    print('划分前后训练样本总数为分别%d,%d' % (df_train.shape[0], trainCount))
    print('划分前后测试样本总数为分别%d,%d' % (df_test.shape[0], testCount))


def decisionTreeGroupDataToList(df_train):
    df_train_fillnan = df_train.copy(deep=True)

    df_train_fillnan['nasrdw_recd_date'].fillna(-99999, inplace=True)
    df_train_fillnan['var_jb_23'].fillna(-99999, inplace=True)
    df_train_fillnan['creditlimitamount_4'].fillna(-99999, inplace=True)
    trainList = []

    trainList.append(
        df_train.loc[(df_train_fillnan['creditlimitamount_4'] <= 299.5) & (df_train_fillnan['var_jb_23'] <= 10.5)]
    )
    trainList.append(
        df_train.loc[(df_train_fillnan['creditlimitamount_4'] <= 299.5) & (df_train_fillnan['var_jb_23'] > 10.5)]
    )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] <= 65954.0) & (
                                 df_train_fillnan['creditlimitamount_4'] > 299.5) & (
                                 df_train_fillnan['nasrdw_recd_date'] <= 20181023.0)]
                     )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] <= 65954.0) & (
                                 df_train_fillnan['creditlimitamount_4'] > 299.5) & (
                                 df_train_fillnan['nasrdw_recd_date'] > 20181023.0)]
                     )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] > 65954.0) & (
                                 df_train_fillnan['creditlimitamount_4'] <= 329720.0) & (
                                 df_train_fillnan['nasrdw_recd_date'] <= 20181023.0)]
                     )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] > 65954.0) & (
                                 df_train_fillnan['creditlimitamount_4'] <= 329720.0) & (
                                 df_train_fillnan['nasrdw_recd_date'] > 20181023.0)]
                     )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] > 329720.0) & (
                                 df_train_fillnan['creditlimitamount_4'] <= 509500.0)]
                     )
    trainList.append(df_train.loc[
                         (df_train_fillnan['creditlimitamount_4'] > 329720.0) & (
                                 df_train_fillnan['creditlimitamount_4'] > 509500.0)]
                     )

    trainCount = 0;

    for i in range(len(trainList)):
        trainCount += trainList[i].shape[0]

    print('划分前后训练样本总数为分别%d,%d' % (df_train.shape[0], trainCount))
    return trainList


# 训练多个模型
def trainMultiModel(trainList, testList, feature_categorical):
    for i in range(len(trainList)):
        df_train, df_test = trainList[i], testList[i]
        print('%d.训练样本%s，测试样本%s' % (i, df_train.shape, df_test.shape))

        X_train, y_train, X_test, y_test = baseline.getTrainTestSample(df_train, df_test, feature_categorical)
        gbm, y_pred = baseline.trainModel(X_train, y_train, X_test, y_test)

        if i == 0:
            all_pred = y_pred
            all_test = y_test
        else:
            all_pred = np.hstack((all_pred, y_pred))
            all_test = np.hstack((all_test, y_test))

        print('The auc score is:', roc_auc_score(all_test, all_pred))

    return all_pred, all_test


def main():
    # df_train, df_test = ParseData.loadPartData()
    df_train, df_test = ParseData.loadData()
    feature_categorical = baseline.getFeatureCategorical(df_train)
    # trainList, testList = descartesGroupDataToList(df_train, df_test, 'nasrdw_recd_date', 'var_jb_28',
    #                                                'var_jb_1', 20181023, 4.5, 23.5)
    # trainList, testList = descartesGroupDataToListTwo(df_train, df_test, 'type_91|个人消费贷款', 0.5,
    #                                                   'var_jb_64', 13.5, 'var_jb_40', 0.5)
    # trainList, testList = descartesGroupDataToListTwo(df_train, df_test, 'creditlimitamount_4', 32188.5,
    #                                                   'var_jb_22', 13.5, 'nasrdw_recd_date', 20181024)
    trainList, testList = descartesGroupDataToListTwo(df_train, df_test, 'var_jb_28', 4.5,
                                                      'var_jb_23', 27.5, 'nasrdw_recd_date', 20181023)
    # trainList = decisionTreeGroupDataToList(df_train)
    # testList = decisionTreeGroupDataToList(df_test)
    all_pred, all_test = trainMultiModel(trainList, testList, feature_categorical)
    Evaluation.getKsValue(all_test, all_pred)
    Evaluation.getAucValue(all_test, all_pred)


if __name__ == '__main__':
    main()
