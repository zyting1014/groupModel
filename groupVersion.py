import standardVersion as baseline
import parseData
"""
    包括多种分群算法的实现
"""

# 分群文件生成
# example:trainList,testList = splitTrainTest
# (df_train,df_test,'nasrdw_recd_date','var_jb_28','var_jb_1',20181023,4.5,22.5)
def splitTrainTest(df_train,df_test,n1,n2,n3,s1,s2,s3):
    df_train_fillnan = df_train.copy()
    df_test_fillnan = df_test.copy()
    df_train[n1].fillna(df_train[n1].mean(),inplace = True)
    df_train[n2].fillna(df_train[n2].mean(),inplace = True)
    df_train[n3].fillna(df_train[n3].mean(),inplace = True)
    df_test[n1].fillna(df_test[n1].mean(),inplace = True)
    df_test[n2].fillna(df_test[n2].mean(),inplace = True)
    df_test[n3].fillna(df_test[n3].mean(),inplace = True)
    trainList = []
    testList = []
    def abc(data,b1,b2,b3):
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
    trainList = [abc(df_train_fillnan,0,0,0),abc(df_train_fillnan,0,0,1),abc(df_train_fillnan,0,1,0),abc(df_train_fillnan,0,1,1),
                 abc(df_train,1,0,0),abc(df_train,1,0,1),abc(df_train,1,1,0),abc(df_train,1,1,1)]
    testList = [abc(df_test_fillnan,0,0,0),abc(df_test_fillnan,0,0,1),abc(df_test_fillnan,0,1,0),abc(df_test_fillnan,0,1,1),
                 abc(df_test_fillnan,1,0,0),abc(df_test_fillnan,1,0,1),abc(df_test_fillnan,1,1,0),abc(df_test_fillnan,1,1,1)]
    print(df_test_fillnan.shape)
    count = 0
    for i in range(8):
        count += testList[i].shape[0]
    print(count)
    return trainList,testList

def main():
    df_train, df_test = parseData.loadPartData()
    feature_categorical = baseline.getFeatureCategorical(df_train)
    trainList, testList = splitTrainTest(df_train, df_test, 'nasrdw_recd_date', 'var_jb_28',
                                         'var_jb_1', 20181023, 4.5,22.5)
    X_train, y_train, X_test, y_test = baseline.getTrainTestSample(df_train, df_test,feature_categorical)
    gbm, y_pred = baseline.trainModel(X_train, y_train, X_test, y_test)
    baseline.Evaluation.getKsValue(X_test, y_test, y_pred)
    baseline.featureImportance(gbm)

if __name__ == '__main__':
    main()