import pandas as pd
import platform
from sklearn.externals import joblib
import os, sys
import numpy as np

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


def loadData_new():
    # 加载你的数据 增加了日期特征和类别特征的处理
    global TYPE
    TYPE = 'origin_new'
    print('Load data...')
    if platform.system() == 'Windows':
        df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2.csv')
        df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2.csv')
    else:
        df_train = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_m2.csv')
        df_test = pd.read_csv('/root/zhuyuting/cxqz/groupModel/data/guidang/data_reserved2.csv')
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
    df_concat = pd.concat([df_train, df_test], axis=0)
    df_train_new = df_concat[df_concat['nasrdw_recd_date'] < 20181215]
    df_test_new = df_concat[df_concat['nasrdw_recd_date'] >= 20181215]
    assert df_concat.shape[0] == df_train_new.shape[0] + df_test_new.shape[0], '划分前后总样本数量不一致！'
    print('新训练集样本数：%d' % df_train_new.shape[0])
    print('新测试集样本数：%d' % df_test_new.shape[0])
    df_train_new.to_csv('归档/data_m2_oot15.csv', index=None)
    df_test_new.to_csv('归档/data_reserved2_oot15.csv', index=None)
    print('保存成功！')


def read_test_result(file_name='result.csv'):
    print('in %s' % sys._getframe().f_code.co_name)
    file_path = path = os.path.join('result', file_name)
    if not os.path.exists(file_path):
        print('结果文件不存在，请检查！')
        return
    result = pd.read_csv(file_path)
    print('读取结果文件..')
    print(result.head())

    y_test = result['true']
    y_pred = result['predict']

    return y_test, y_pred


def save_test_result(y_test, y_pred):
    # 传入的为list类型
    # 测试集结果存为csv 有两列 分别是预测值，实际值
    # 命名规则为运行的主类加...
    print('in %s' % sys._getframe().f_code.co_name)
    file_name = 'result.csv'
    result = pd.DataFrame(np.array([list(y_pred), list(y_test)]).T, columns=['true', 'predict'])
    result.to_csv('result/result.csv', index=False)
    print('保存结果到%s..' % file_name)
