import numpy as np
from sklearn.linear_model import LinearRegression
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
        print('%f' % (a / (b + 0.0001)), end='#')
    print('\n划分前后训练样本总数为分别%d,%d' % (df_train.shape[0], train_count))
    print('划分前后测试样本总数为分别%d,%d' % (df_test.shape[0], test_count))

    assert df_train.shape[0] == train_count, '训练集样本划分前后数量不一致'
    assert df_test.shape[0] == test_count, '测试集样本划分前后数量不一致'
    assert len(train_list) == len(test_list), '训练测试集类别数不一致'


# 将多列已经编好的one-hot转为多个模型
def transOnehotToList(df_train, df_test, one_hot_list):
    if len(one_hot_list) == 0:
        return [df_train], [df_test]

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
    all_pred = []
    pred_list = []
    true_list = []
    for i in range(len(train_list)):
        df_train, df_test = train_list[i], test_list[i]
        print('%d.训练样本%s，测试样本%s' % (i, df_train.shape, df_test.shape))
        # 测试集该类没有样本 跳过
        if df_test.shape[0] == 0:
            continue

        X_train, y_train, X_test, y_test = baseline.getTrainTestSample(df_train, df_test, feature_categorical)

        gbm, y_pred = baseline.trainModel(X_train, y_train, X_test, y_test)
        model_list.append(gbm)

        pred_list.append(y_pred)
        true_list.append(y_test)

        if len(all_pred) == 0:
            all_pred = y_pred
            all_test = y_test
        else:
            all_pred = np.hstack((all_pred, y_pred))
            all_test = np.hstack((all_test, y_test))

        print('The auc score is:', roc_auc_score(all_test, all_pred))

    return all_pred, all_test, model_list, pred_list, true_list


def supervised_method(df_train, df_test):
    import GroupFunc
    column_name = []
    # implement code here

    return df_train, df_test, column_name

def unsupervised_method(df_train, df_test):
    import GroupFunc
    column_name = []
    # df_train, df_test, column_name = GroupFunc.getGMMCategoryFeature(df_train, df_test, feature_categorical, 4)
    # train_list, test_list = transOnehotToList(df_train, df_test, column_name)

    # df_train, df_test, column_name = GroupFunc.getGMMCategoryFeature(df_train, df_test, 2)

    df_train, df_test, column_name = GroupFunc.getGMMNullFeature(df_train, df_test, 4)


    # df_train, df_test, column_name = GroupFunc.getKmeansAllFeature(df_train, df_test, 3)

    # df_train, df_test, column_name = GroupFunc.getKmeansNullFeature(df_train, df_test, 2)

    # df_train, df_test, column_name = GroupFunc.nullCountcut(df_train, df_test)

    return df_train, df_test, column_name


def transSampleToList(df_train, df_test, feature_categorical):
    import GroupFunc_old

    train_list, test_list = [df_train], [df_test]

    df_train, df_test, column_name = supervised_method(df_train, df_test)
    df_train, df_test, column_name = unsupervised_method(df_train, df_test)

    train_list, test_list = transOnehotToList(df_train, df_test, column_name)
    pos_rate_in_each_segment(train_list, test_list)

    return train_list, test_list


def pos_rate_in_each_segment(train_list, test_list):

    for i in range(len(train_list)):
        train_sample = train_list[i]
        test_sample = test_list[i]
        true_pos_train = train_sample[train_sample['bad'] == 1].shape[0]
        true_neg_train = train_sample[train_sample['bad'] == 0].shape[0]
        true_pos_test = test_sample[test_sample['bad'] == 1].shape[0]
        true_neg_test = test_sample[test_sample['bad'] == 0].shape[0]

        print('在训练实际的样本中，%d为正样本，%d为负样本，正负比例为%f' % (true_pos_train, true_neg_train, (true_pos_train / true_neg_train)))
        print('在测试实际的样本中，%d为正样本，%d为负样本，正负比例为%f' % (true_pos_test, true_neg_test, (true_pos_test / true_neg_test)))


# 分数校正
def fractional_calibration(pred_list, true_list):
    new_pred_list = []
    new_pred = []
    i = 0
    for item in zip(pred_list, true_list):
        pred = item[0].reshape(-1, 1)
        true = np.array(item[1])

        lg = LinearRegression()
        model = lg.fit(pred, true)
        # print(model.coef_)  # 斜率 k
        # print(model.intercept_)  # 常数b
        k = str(model.coef_[0])
        b = '+' + str(model.intercept_) if model.intercept_ > 0 else str(model.intercept_)
        print('群体%d正在进行分数矫正..矫正式为y=%sx%s' % (i, k, b))
        i = i + 1
        pred_ = model.predict(pred)

        new_pred_list.append(pred_)

        if len(new_pred_list) == 0:
            new_pred = pred_
        else:
            new_pred = np.hstack((new_pred, pred_))

    return new_pred, new_pred_list


def main():
    # df_train, df_test = ParseData.loadPartData()
    # df_train, df_test = ParseData.loadData()
    # df_train, df_test = ParseData.loadOOTData()
    # df_train, df_test = ParseData.loadOOT15Data()
    df_train, df_test = ParseData.loadData_new()

    # ParseData.TYPE = 'OOT_noDate'

    ##################################################
    # 日期特征处理
    feature_date = baseline.getFeatureDate(df_train)
    df_train = baseline.parseDateToInt(df_train, feature_date)
    df_test = baseline.parseDateToInt(df_test, feature_date)
    # 类别特征处理
    df_train, df_test = baseline.CategoryPCA(df_train, df_test, baseline.getFeatureCategorical(df_train))
    ##################################################

    feature_categorical = baseline.getFeatureCategorical(df_train)

    train_list, test_list = transSampleToList(df_train, df_test, feature_categorical)

    all_pred, all_true, model_list, pred_list, true_list = trainMultiModel(train_list, test_list, feature_categorical)

    new_pred, new_pred_list = fractional_calibration(pred_list, true_list) # 分数校准
    all_pred = new_pred

    Evaluation.getKsValue(all_true, all_pred)
    Evaluation.getAucValue(all_true, all_pred)

    Evas.main(model_list)
    Evaluation.get_pos_neg_picture(all_true, all_pred)


if __name__ == '__main__':
    main()
