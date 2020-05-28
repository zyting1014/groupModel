import ParseData
import StandardVersion as baseline
from xgboost import XGBClassifier
import xgboost
import sys
import CodeGenerateXgb
'''
    graphiz的view函数中文乱码的问题有待解决,不加view显示正确
'''


# 只取重要特征训练
def getTrainTestSampleImportance(df_train, df_test, feature_categorical, importance_list):
    target = 'bad'
    feature_categorical.append(target)
    # y_train = pd.get_dummies(df_train, columns=feature_categorical)
    # X_train = pd.get_dummies(df_test, columns=feature_categorical)
    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train[importance_list]
    X_test = df_test[importance_list]

    # X_train = X_train.drop(feature_categorical, axis=1)
    # X_test = X_test.drop(feature_categorical, axis=1)

    return X_train, y_train, X_test, y_test


def getTrainTestSample(df_train, df_test, feature_categorical):
    print('in %s' % sys._getframe().f_code.co_name)
    target = 'bad'
    feature_categorical.append(target)
    # y_train = pd.get_dummies(df_train, columns=feature_categorical)
    # X_train = pd.get_dummies(df_test, columns=feature_categorical)
    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train
    X_test = df_test
    X_train = df_train.drop(feature_categorical, axis=1)
    X_test = df_test.drop(feature_categorical, axis=1)
    X_train = X_train.drop('nasrdw_recd_date', axis=1)
    X_test = X_test.drop('nasrdw_recd_date', axis=1)

    return X_train, y_train, X_test, y_test


def trainModel(X_train, y_train, X_test, y_test, round=5):
    print('in %s' % sys._getframe().f_code.co_name)
    xgb = XGBClassifier(nthread=-1, n_estimators=round)
    eval_set = [(X_test, y_test)]
    xgb.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=eval_set, verbose=True)
    # xgb.save_model('xgboost.model')
    # y_pred = xgb.predict(X_test)
    # print('The auc score is:', roc_auc_score(y_test, y_pred))
    return xgb


def showPicture(xgb, num_trees=0):
    print('in %s' % sys._getframe().f_code.co_name)
    xgboost.to_graphviz(xgb, num_trees=0)
    img = xgboost.to_graphviz(xgb, num_trees=num_trees)  # 这行直接可以放在jupyter理正确显示
    img.format = 'png'
    img.view('image//xgb')
    return img




def main():
    # df_train, df_test = ParseData.loadPartData()
    df_train, df_test = ParseData.loadData()

    feature_categorical = baseline.getFeatureCategorical(df_train)

    ##################################################
    # 日期特征处理
    feature_date = baseline.getFeatureDate(df_train)
    df_train = baseline.parseDateToInt(df_train, feature_date)
    df_test = baseline.parseDateToInt(df_test, feature_date)
    ##################################################

    importance_list = ['nasrdw_recd_date', 'var_jb_43', 'var_jb_94', 'creditlimitamount_4',
                      'var_jb_15', 'var_jb_23', 'creditlimitamount_3', 'var_jb_73', 'var_jb_25', 'var_jb_22']

    # importance_list = ['querydate', 'nasrdw_recd_date', 'var_jb_43', 'opendate_3', 'var_jb_94', 'opendate_4',
    #  'var_jb_23', 'var_jb_15', 'creditlimitamount_4', 'var_jb_73']

    X_train, y_train, X_test, y_test = getTrainTestSampleImportance(df_train, df_test, feature_categorical, importance_list)

    xgb = trainModel(X_train, y_train, X_test, y_test)

    img = showPicture(xgb, 0)

    train_sentence, test_sentence, seg_sentence = CodeGenerateXgb.create(img)
    print(train_sentence)
    print(test_sentence)
    print(seg_sentence)




if __name__ == '__main__':
    main()