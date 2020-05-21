import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sys


def getAucValue(y_test, y_pred):
    print('The auc score is:', roc_auc_score(y_test, y_pred))


def getKsValue(y_test, y_pred):
    df = pd.DataFrame()
    df['y_xgb_test'] = y_test
    df['y_xgb_pred'] = y_pred

    kstable = ks(df, 'y_xgb_test', 'y_xgb_pred')
    get_pos_neg_cnt(y_test, y_pred, get_predict_point(kstable))


def ks(data=None, target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events'] = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate'] = (kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100

    # Formating
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1, 11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)

    # Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS'])) + "%" + " at decile " + str(
        (kstable.index[kstable['KS'] == max(kstable['KS'])][0])))

    return kstable


# 返回正负预测 array
def get_pos_neg_predict(y_test, y_pred):
    print('in %s' % sys._getframe().f_code.co_name)
    y = pd.DataFrame(np.array([list(y_test), list(y_pred)]).T, columns=['true', 'pred'])
    pos_pred = y[y['true'] == 1]['pred']
    neg_pred = y[y['true'] == 0]['pred']
    print(pos_pred.shape)
    print(neg_pred.shape)
    return pos_pred, neg_pred


def draw_pos_neg_picture(pos_pred, neg_pred):
    print('in %s' % sys._getframe().f_code.co_name)
    # 使正负样本量一致 用nan填充
    pos_cnt = pos_pred.shape[0]
    neg_cnt = neg_pred.shape[0]
    if neg_cnt > pos_cnt:
        pos_pred_fill = np.array([np.nan] * (neg_cnt - pos_cnt))
        pos_pred_fill = np.hstack((pos_pred, pos_pred_fill))
        neg_pred_fill = neg_pred
    else:
        neg_pred_fill = np.array([np.nan] * (pos_cnt - neg_cnt))
        neg_pred_fill = np.hstack((neg_pred, neg_pred_fill))
        pos_pred_fill = pos_pred

    dist = pd.DataFrame(np.array([pos_pred_fill, neg_pred_fill]).T, columns=['good', 'bad'])
    #     dist= pd.DataFrame(np.array([[0.1,0.2,0.2,np.nan],[0.7,0.8,0.8,0.9]]).T,columns=['a','b'])
    fig, ax = plt.subplots(figsize=(15, 6))
    dist.plot.kde(ax=ax, legend=False, title='Histogram: good vs. bad')
    dist.plot.hist(density=False, ax=ax, color=['red', 'blue'], histtype='barstacked')  # density=Frue用于加轮廓
    ax.set_ylabel('Frequency')
    ax.grid(axis='sample')
    ax.set_facecolor('#d8dcd6')
    plt.xlim(0, 1)
    plt.show()


# 作结果分布图 调用save_test_result、get_test_result、get_pos_neg_predict、draw_pos_neg_picture等函数
def get_pos_neg_picture(y_test, y_pred):
    import ParseData
    print('in %s' % sys._getframe().f_code.co_name)
    ParseData.save_test_result(y_test, y_pred)
    y_test, y_pred = ParseData.read_test_result()
    pos_pred, neg_pred = get_pos_neg_predict(y_pred, y_test)
    draw_pos_neg_picture(pos_pred, neg_pred)


# 获得预测划分点（ks最大的点的概率值）
def get_predict_point(kstable):
    max_decile = kstable.index[kstable['KS'] == max(kstable['KS'])][0]
    predict_point = kstable['min_prob'][max_decile]

    return predict_point


# 获得正负样本真实、预测个数 传入为array或Series
def get_pos_neg_cnt(y_true, y_pred, predict_point):
    df = pd.DataFrame()
    df['true'] = y_true
    df['pred'] = y_pred
    true_pos = df[df['true'] == 1].shape[0]
    true_neg = df[df['true'] == 0].shape[0]
    pred_pos = df[df['pred'] > predict_point].shape[0]  # ks是左开右闭区间
    pred_neg = df[df['pred'] <= predict_point].shape[0]
    assert true_pos + true_neg == pred_pos + pred_neg, '真值和预测值的数量不一致！'
    print('在实际的样本中，%d为正样本，%d为负样本，正负比例为%f' % (true_pos, true_neg, (true_pos / true_neg)))
    print('在预测的样本中，%d为正样本，%d为负样本，正负比例为%f' % (pred_pos, pred_neg, (pred_pos / pred_neg)))
