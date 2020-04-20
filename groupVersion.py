import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture as GMM
import standardVersion as baseline
import parseData
import Evaluation

"""
    包括多种分群算法的实现
"""


# 分群文件生成 生成8个训练、测试集
# example:trainList,testList = splitTrainTest
# (df_train,df_test,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def splitTrainTest(df_train, df_test, n1, n2, n3, s1, s2, s3):
    df_train_fillnan = df_train.copy()
    df_test_fillnan = df_test.copy()
    df_train[n1].fillna(df_train[n1].mean(), inplace=True)
    df_train[n2].fillna(df_train[n2].mean(), inplace=True)
    df_train[n3].fillna(df_train[n3].mean(), inplace=True)
    df_test[n1].fillna(df_test[n1].mean(), inplace=True)
    df_test[n2].fillna(df_test[n2].mean(), inplace=True)
    df_test[n3].fillna(df_test[n3].mean(), inplace=True)
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
                 abc(df_train, 1, 0, 0), abc(df_train, 1, 0, 1), abc(df_train, 1, 1, 0), abc(df_train, 1, 1, 1)]
    testList = [abc(df_test_fillnan, 0, 0, 0), abc(df_test_fillnan, 0, 0, 1), abc(df_test_fillnan, 0, 1, 0),
                abc(df_test_fillnan, 0, 1, 1),
                abc(df_test_fillnan, 1, 0, 0), abc(df_test_fillnan, 1, 0, 1), abc(df_test_fillnan, 1, 1, 0),
                abc(df_test_fillnan, 1, 1, 1)]
    print(df_test_fillnan.shape)
    count = 0
    for i in range(8):
        count += testList[i].shape[0]
    print(count)
    return trainList, testList


# 分群作为新特征 8列
# example:splitTrainTest2(df_train,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def splitTrainTest2(df_train, n1, n2, n3, s1, s2, s3):
    df_train['seg1'] = 0;
    df_train['seg2'] = 0;
    df_train['seg3'] = 0;
    df_train['seg4'] = 0
    df_train['seg5'] = 0;
    df_train['seg6'] = 0;
    df_train['seg7'] = 0;
    df_train['seg8'] = 0
    df_train.loc[(df_train[n1] < s1) & (df_train[n2] < s2) & (df_train[n3] < s3), 'seg1'] = 1
    df_train.loc[(df_train[n1] < s1) & (df_train[n2] < s2) & (df_train[n3] >= s3), 'seg2'] = 1
    df_train.loc[(df_train[n1] < s1) & (df_train[n2] >= s2) & (df_train[n3] < s3), 'seg3'] = 1
    df_train.loc[(df_train[n1] < s1) & (df_train[n2] >= s2) & (df_train[n3] >= s3), 'seg4'] = 1
    df_train.loc[(df_train[n1] >= s1) & (df_train[n2] < s2) & (df_train[n3] < s3), 'seg5'] = 1
    df_train.loc[(df_train[n1] >= s1) & (df_train[n2] < s2) & (df_train[n3] >= s3), 'seg6'] = 1
    df_train.loc[(df_train[n1] >= s1) & (df_train[n2] >= s2) & (df_train[n3] < s3), 'seg7'] = 1
    df_train.loc[(df_train[n1] >= s1) & (df_train[n2] >= s2) & (df_train[n3] >= s3), 'seg8'] = 1


# 高斯混合模型
def getGMMFeature(df_train, df_test, feature_categorical, n_components=4):
    X_train = df_train.drop(feature_categorical, axis=1).copy().fillna(0)
    X_test = df_test.drop(feature_categorical, axis=1).copy().fillna(0)
    gmm = GMM(n_components=n_components).fit(X_train)
    labels_train = gmm.predict(X_train)
    labels_test = gmm.predict(X_test)
    df_train['gmm'] = labels_train.tolist()
    df_test['gmm'] = labels_test.tolist()

    return df_train,df_test


# 训练8个模型
def trainMultiModel(trainList, testList, feature_categorical):
    for i in range(8):
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
    # df_train, df_test = parseData.loadPartData()
    df_train, df_test = parseData.loadData()
    feature_categorical = baseline.getFeatureCategorical(df_train)
    trainList, testList = splitTrainTest(df_train, df_test, 'nasrdw_recd_date', 'var_jb_28',
                                         'var_jb_1', 20181023, 4.5, 22.5)
    all_pred, all_test = trainMultiModel(trainList, testList, feature_categorical)
    Evaluation.getKsValue(all_test, all_pred)
    Evaluation.getAucValue(all_test, all_pred)


if __name__ == '__main__':
    main()
