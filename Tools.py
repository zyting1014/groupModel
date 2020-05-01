"""
    工具类
"""
import numpy as np
import pandas.api.types as types

# iv_more_than_point_one与feature_categorical互斥
iv_more_than_point_one = ["nasrdw_recd_date", "var_jb_20", "var_jb_21", "var_jb_24", "var_jb_28", "var_jb_33", "var_jb_34", "var_jb_40",
      "var_jb_75", "var_jb_76", "var_jb_79", "var_jb_88", "var_jb_89", "var_jb_91", "var_jb_92",
      "curroverdueamount_3", "creditlimitamount_4", "balance", "remainpaymentcyc", "scheduledpaymentamount_4",
      "actualpaymentamount_4", "curroverduecyc_4", "curroverdueamount_4", "overdue31to60amount_7",
      "overdue61to90amount_7", "overdue91to180amount_7", "creditlimitamount", "overdueover180amount_8",
      "latest5yearoverduebeginmonth", "latest5yearoverdueendmonth", "var_jb_4_O", "guaranteetype_3_1|质押（含保证金）",
      "guaranteetype_3_3|保证", "currency_3_澳门元", "financetype_3_住房储蓄银行", "financetype_3_外资银行", "paymentrating_02|周",
      "guaranteetype_4_4|信用/免担保", "type_91|个人消费贷款", "financetype_4_住房公积金管理中心", "financetype_4_消费金融有限公司",
      "class5state_5|损失", "var_jb_48_专升本", "var_jb_48_博士研究生", "var_jb_48_夜大电大函大普通班", "var_jb_55_不详",
      "var_jb_55_夜大学", "var_jb_47_全日制", "var_jb_47_夜大学", "var_jb_47_研究生", "var_jb_54_全日制", "var_jb_54_夜大学",
      "var_jb_54_研究生", "guaranteetype_1|质押（含保证金）", "guaranteetype_6|组合（不含保证）", "state_3|止付", "state_5|呆帐",
      "var_jb_39_B", "var_jb_39_D", "var_jb_39_K", "var_jb_39_L", "var_jb_93_Z"]

feature_categorical = ['id', 'var_jb_2', 'var_jb_3', 'var_jb_16', 'var_jb_17', 'var_jb_45',
                       'var_jb_46', 'var_jb_53', 'var_jb_81', 'var_jb_85', 'ctime_3', 'financeorg_3',
                       'opendate_3', 'stateenddate_3', 'scheduledpaymentdate_3', 'recentpaydate_3', 'latest24state_3', 'ctime_4',
                       'financeorg_4', 'opendate_4', 'enddate', 'paymentcyc', 'stateenddate_4', 'scheduledpaymentdate_4',
                       'recentpaydate_4', 'latest24state_4', 'ctime_7', 'querydate', 'querier', 'ctime_8',
                       'financeorg', 'opendate', 'stateenddate', 'scheduledpaymentdate', 'recentpaydate','latest24state']


def apply_log1p_transformation(dataframe, columns):
    for column in columns:
        if types.is_int64_dtype(dataframe[column]):
            dataframe[column] = np.log1p(dataframe[column])

    return dataframe
