import pandas as pd

def loadData():
    # 加载你的数据
    print('Load data...')
    df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2.csv')
    df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2.csv')
    return df_train, df_test

def loadPartData():
    # 加载你的数据
    print('Load data...')
    df_train = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_m2_part.csv')
    df_test = pd.read_csv('D:/zhuyuting/cxqz/task1/归档/data_reserved2.csv')
    return df_train, df_test
