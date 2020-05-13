import pandas as pd
import platform
from sklearn.externals import joblib
import os
TYPE = 'origin'


def loadData():
    # 加载你的数据
    print('Load data...')
    if platform.system() == 'Windows':
        df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2.csv')
        df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2.csv')
    else:
        df_train = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_m2.csv')
        df_test = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_reserved2.csv')
    return df_train, df_test


def loadPartData():
    # 加载你的数据
    print('Load data...')
    if platform.system() == 'Windows':
        df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2_part.csv')
        df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2.csv')
    else:
        df_train = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_m2_part.csv')
        df_test = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_reserved2.csv')
    return df_train, df_test


def loadOOTData():
    # 加载你的数据
    print('Load oot data...')
    global TYPE
    TYPE = 'OOT'
    print('系统为%s系统..' % platform.system())
    if platform.system() == 'Windows':
        df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2_oot.csv')
        df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2_oot.csv')
    else:
        df_train = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_m2_oot.csv')
        df_test = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_reserved2_oot.csv')
    return df_train, df_test


def loadOOT15Data():
    # 加载你的数据
    print('Load oot15 data...')
    global TYPE
    TYPE = 'OOT15'
    print('系统为%s系统..' % platform.system())
    if platform.system() == 'Windows':
        df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2_oot15.csv')
        df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2_oot15.csv')
    else:
        df_train = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_m2_oot15.csv')
        df_test = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_reserved2_oot15.csv')
    return df_train, df_test


def saveModel(model, model_name):
    path = os.path.join('model', TYPE, model_name)
    print(path)
    joblib.dump(model, path)


def loadModel(model_name):
    path = os.path.join('model', TYPE, model_name)
    print(path)
    return joblib.load(path)


def existModel(model_name):
    path = os.path.join('model', TYPE, model_name)
    print(path)
    return os.path.exists(path)


def splitOOT(df_train, df_test):
    print('原训练集样本数：%d' % df_train.shape[0])
    print('原测试集样本数：%d' % df_test.shape[0])
    df_concat = pd.concat([df_train, df_test], axis=0)
    df_train_new = df_concat[df_concat['nasrdw_recd_date'] < 20181201]
    df_test_new = df_concat[df_concat['nasrdw_recd_date'] >= 20181201]
    assert df_concat.shape[0] == df_train_new.shape[0] + df_test_new.shape[0], '划分前后总样本数量不一致！'
    print('新训练集样本数：%d' % df_train_new.shape[0])
    print('新测试集样本数：%d' % df_test_new.shape[0])
    df_train_new.to_csv('归档/data_m2_oot.csv', index=None)
    df_test_new.to_csv('归档/data_reserved2_oot.csv', index=None)
    print('保存成功！')


def splitOOT15(df_train, df_test):
    print('原训练集样本数：%d' % df_train.shape[0])
    print('原测试集样本数：%d' % df_test.shape[0])
    df_concat = pd.concat([df_train, df_test],axis = 0)
    df_train_new = df_concat[df_concat['nasrdw_recd_date'] < 20181215]
    df_test_new = df_concat[df_concat['nasrdw_recd_date'] >= 20181215]
    assert df_concat.shape[0] == df_train_new.shape[0] + df_test_new.shape[0], '划分前后总样本数量不一致！'
    print('新训练集样本数：%d' % df_train_new.shape[0])
    print('新测试集样本数：%d' % df_test_new.shape[0])
    df_train_new.to_csv('归档/data_m2_oot15.csv',index=None)
    df_test_new.to_csv('归档/data_reserved2_oot15.csv',index=None)
    print('保存成功！')


