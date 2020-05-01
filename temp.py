import numpy as np
from sklearn.metrics import roc_auc_score
import StandardVersion as baseline
import ParseData
import Evaluation
import EvaluateSegmentation as Evas

"""
    实现同时训练多个模型
"""


# 分群文件生成 生成2个训练、测试集
def treeGroupDataToListOne(df_train, df_test, n1, s1, onehot=False):
    df_train_fillnan = df_train.copy(deep=True)
    df_test_fillnan = df_test.copy(deep=True)
    df_train_fillnan[n1].fillna(-99999, inplace=True)
    df_test_fillnan[n1].fillna(-99999, inplace=True)
    if not onehot:
        train_list = [df_train[df_train_fillnan[n1] < s1], df_train[df_train_fillnan[n1] >= s1]]
        test_list = [df_test[df_test_fillnan[n1] < s1], df_test[df_test_fillnan[n1] >= s1]]
    else:
        train_list = [df_train[df_train_fillnan[n1] == s1], df_train[df_train_fillnan[n1] != s1]]
        test_list = [df_test[df_test_fillnan[n1] == s1], df_test[df_test_fillnan[n1] != s1]]

    checkTrainTestList(df_train, df_test, train_list, test_list)

    return train_list, test_list


# 分群文件生成 生成3个训练、测试集
def descartesGroupDataToListOneHalf(df_train, df_test, n1, s1, c1, cs1):
    train_list = [
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & ((df_train[c1] < cs1) | (df_train[c1].isnull()))],
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & (df_train[c1] >= cs1)],
        df_train[(df_train[n1] >= s1)]
    ]
    test_list = [
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & ((df_test[c1] < cs1) | (df_test[c1].isnull()))],
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & (df_test[c1] >= cs1)],
        df_test[(df_test[n1] >= s1)]
    ]
    checkTrainTestList(df_train, df_test, train_list, test_list)

    return train_list, test_list


# 分群文件生成 生成4个训练、测试集
def treeGroupDataToListTwo(df_train, df_test, n1, s1, c1, cs1, c2, cs2):
    print('in treeGroupDataToListTwo..')
    train_list = [
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & ((df_train[c1] < cs1) | (df_train[c1].isnull()))],
        df_train[((df_train[n1] < s1) | (df_train[n1].isnull())) & (df_train[c1] >= cs1)],
        df_train[(df_train[n1] >= s1) & ((df_train[c2] < cs2) | (df_train[c2].isnull()))],
        df_train[(df_train[n1] >= s1) & (df_train[c2] >= cs2)],
    ]
    test_list = [
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & ((df_test[c1] < cs1) | (df_test[c1].isnull()))],
        df_test[((df_test[n1] < s1) | (df_test[n1].isnull())) & (df_test[c1] >= cs1)],
        df_test[(df_test[n1] >= s1) & ((df_test[c2] < cs2) | (df_test[c2].isnull()))],
        df_test[(df_test[n1] >= s1) & (df_test[c2] >= cs2)],
    ]

    checkTrainTestList(df_train, df_test, train_list, test_list)

    return train_list, test_list


# 分群文件生成 生成8个训练、测试集
def descartesGroupDataToListThree(df_train, df_test, n1, n2, n3, s1, s2, s3):
    df_train_fillnan = df_train.copy(deep=True)
    df_test_fillnan = df_test.copy(deep=True)
    df_train_fillnan[n1].fillna(-99999, inplace=True)
    df_train_fillnan[n2].fillna(-99999, inplace=True)
    df_train_fillnan[n3].fillna(-99999, inplace=True)
    df_test_fillnan[n1].fillna(-99999, inplace=True)
    df_test_fillnan[n2].fillna(-99999, inplace=True)
    df_test_fillnan[n3].fillna(-99999, inplace=True)
    train_list = []
    test_list = []

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

    train_list = [abc(df_train_fillnan, 0, 0, 0), abc(df_train_fillnan, 0, 0, 1), abc(df_train_fillnan, 0, 1, 0),
                 abc(df_train_fillnan, 0, 1, 1),
                 abc(df_train_fillnan, 1, 0, 0), abc(df_train_fillnan, 1, 0, 1), abc(df_train_fillnan, 1, 1, 0),
                 abc(df_train_fillnan, 1, 1, 1)]
    test_list = [abc(df_test_fillnan, 0, 0, 0), abc(df_test_fillnan, 0, 0, 1), abc(df_test_fillnan, 0, 1, 0),
                abc(df_test_fillnan, 0, 1, 1),
                abc(df_test_fillnan, 1, 0, 0), abc(df_test_fillnan, 1, 0, 1), abc(df_test_fillnan, 1, 1, 0),
                abc(df_test_fillnan, 1, 1, 1)]

    checkTrainTestList(df_train, df_test, test_list, test_list)

    return train_list, test_list


# 检查划分前后样本个数是否一致
def checkTrainTestList(df_train, df_test, train_list, test_list):
    train_count = 0
    test_count = 0
    each_group_count = []
    print('分群数量为%d' % (len(train_list)))
    for i in range(len(train_list)):
        train_count += train_list[i].shape[0]
        test_count += test_list[i].shape[0]
        each_group_count.append((train_list[i].shape[0], test_list[i].shape[0]))
    print(each_group_count)
    print('类别数为%d' % len(each_group_count))
    for (a, b) in each_group_count:
        print('%f' % (a/(b+0.0001)), end='#')
    print('\n划分前后训练样本总数为分别%d,%d' % (df_train.shape[0], train_count))
    print('划分前后测试样本总数为分别%d,%d' % (df_test.shape[0], test_count))

    assert df_train.shape[0] == train_count, '训练集样本划分前后数量不一致'
    assert df_test.shape[0] == test_count, '测试集样本划分前后数量不一致'
    assert len(train_list) == len(test_list), '训练测试集类别数不一致'


# 将多列已经编好的one-hot转为多个模型
def transOnehotToList(df_train, df_test, one_hot_list):
    train_list = []
    test_list = []
    for column in one_hot_list:
        train_list.append(df_train[df_train[column] == 1])
        test_list.append(df_test[df_test[column] == 1])
    checkTrainTestList(df_train, df_test, train_list, test_list)
    return train_list, test_list


# 训练多个模型
def trainMultiModel(train_list, test_list, feature_categorical):
    model_list = []
    for i in range(len(train_list)):
        df_train, df_test = train_list[i], test_list[i]
        print('%d.训练样本%s，测试样本%s' % (i, df_train.shape, df_test.shape))

        X_train, y_train, X_test, y_test = baseline.getTrainTestSample(df_train, df_test, feature_categorical)

        gbm, y_pred = baseline.trainModel(X_train, y_train, X_test, y_test)
        model_list.append(gbm)

        if i == 0:
            all_pred = y_pred
            all_test = y_test
        else:
            all_pred = np.hstack((all_pred, y_pred))
            all_test = np.hstack((all_test, y_test))

        print('The auc score is:', roc_auc_score(all_test, all_pred))

    return all_pred, all_test, model_list


def transSampleToList(df_train, df_test, feature_categorical):
    import GroupFunc
    train_list, test_list = [df_train], [df_test]

    # trainList, testList = descartesGroupDataToListThree(df_train, df_test, 'nasrdw_recd_date', 'var_jb_28',
    #                                                'var_jb_1', 20181023, 4.5, 23.5)
    # trainList, testList = treeGroupDataToListTwo(df_train, df_test, 'type_91|个人消费贷款', 0.5,
    #                                                   'var_jb_64', 13.5, 'var_jb_40', 0.5)
    # train_list, test_list = treeGroupDataToListTwo(df_train, df_test, 'creditlimitamount_4', 32188.5,
    #                                                   'var_jb_22', 13.5, 'nasrdw_recd_date', 20181024)
    # trainList, testList = treeGroupDataToListTwo(df_train, df_test, 'var_jb_28', 4.5,
    #                                                   'var_jb_23', 27.5, 'nasrdw_recd_date', 20181023)
    #######################################################################################
    # df_train, df_test, column_name = GroupFunc.getGMMCategoryFeature(df_train, df_test, feature_categorical, 4)
    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    # df_train, column_name = GroupFunc.decisionTreeMethod2(df_train)
    # df_test, column_name = GroupFunc.decisionTreeMethod2(df_test)
    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    # df_train, df_test, column_name = GroupFunc.getGMMNullFeature(df_train, df_test, 2)
    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    # df_train, column_name = GroupFunc.decisionTreeMethod2(df_train)
    # df_test, column_name = GroupFunc.decisionTreeMethod2(df_test)



    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    # df_train, df_test, column_name = GroupFunc.nullCountcut(df_train, df_test)
    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)


    df_train, df_test, column_name = GroupFunc.getKmeansAllFeature(df_train, df_test, 3)
    train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    return train_list, test_list


def main():
    # df_train, df_test = ParseData.loadPartData()
    df_train, df_test = ParseData.loadData()
    feature_categorical = baseline.getFeatureCategorical(df_train)

    train_list, test_list = transSampleToList(df_train, df_test, feature_categorical)

    all_pred, all_test, model_list = trainMultiModel(train_list, test_list, feature_categorical)
    Evaluation.getKsValue(all_test, all_pred)
    Evaluation.getAucValue(all_test, all_pred)

    Evas.main(model_list)


if __name__ == '__main__':
    main()
