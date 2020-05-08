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

not_feature_categorical = ['nasrdw_recd_date','var_jb_1','var_jb_7','var_jb_9','var_jb_11','var_jb_12',
                            'var_jb_13','var_jb_14','var_jb_15','var_jb_19','var_jb_20','var_jb_21',
                            'var_jb_22','var_jb_23','var_jb_24','var_jb_25','var_jb_26','var_jb_27',
                            'var_jb_28','var_jb_29','var_jb_30','var_jb_31','var_jb_32','var_jb_33',
                            'var_jb_34','var_jb_35','var_jb_36','var_jb_37','var_jb_38','var_jb_40',
                            'var_jb_41','var_jb_42','var_jb_43','var_jb_44','var_jb_50','var_jb_51',
                            'var_jb_52','var_jb_56','var_jb_58','var_jb_59','var_jb_60','var_jb_61',
                            'var_jb_63','var_jb_64','var_jb_65','var_jb_66','var_jb_67','var_jb_68',
                            'var_jb_69','var_jb_70','var_jb_71','var_jb_72','var_jb_73','var_jb_74',
                            'var_jb_75','var_jb_76','var_jb_77','var_jb_78','var_jb_79','var_jb_80',
                            'var_jb_82','var_jb_83','var_jb_84','var_jb_86','var_jb_87','var_jb_88',
                            'var_jb_89','var_jb_90','var_jb_91','var_jb_92','var_jb_94','var_jb_95',
                            'var_jb_96','var_jb_97','var_jb_98','index_3','creditlimitamount_3','sharecreditlimitamount_7',
                            'usedcreditlimitamount_7','latest6monthusedavgamount_7','usedhighestamount_7','scheduledpaymentamount_3','actualpaymentamount_3','curroverduecyc_3',
                            'curroverdueamount_3','latest24monthpaymentbeginmonth_3','latest24monthpaymentendmonth_3','latest5yearoverduebeginmonth_3','latest5yearoverdueendmonth_3','mis_date_7',
                            'index_4','creditlimitamount_4','stateendmonth_7','balance','remainpaymentcyc','scheduledpaymentamount_4',
                            'actualpaymentamount_4','curroverduecyc_4','curroverdueamount_4','overdue31to60amount_7','overdue61to90amount_7','overdue91to180amount_7',
                            'overdueover180amount_7','latest24monthpaymentbeginmonth_4','latest24monthpaymentendmonth_4','latest5yearoverduebeginmonth_4','latest5yearoverdueendmonth_4','parentindex',
                            'index','creditlimitamount','stateendmonth_8','sharecreditlimitamount_8','usedcreditlimitamount_8','latest6monthusedavgamount_8',
                            'usedhighestamount_8','scheduledpaymentamount','actualpaymentamount','curroverduecyc','curroverdueamount','overdue31to60amount_8',
                            'overdue61to90amount_8','overdue91to180amount_8','overdueover180amount_8','latest24monthpaymentbeginmonth','latest24monthpaymentendmonth','latest5yearoverduebeginmonth',
                            'latest5yearoverdueendmonth','mis_date_8','var_jb_4_H','var_jb_4_M','var_jb_4_O','var_jb_4_P',
                            'var_jb_4_S','var_jb_4_U','var_jb_5_F','var_jb_5_M','var_jb_8_M','var_jb_8_O',
                            'var_jb_8_S','var_jb_10_B','var_jb_10_H','var_jb_18_A','var_jb_18_B','var_jb_18_C',
                            'var_jb_18_D','var_jb_18_E','var_jb_18_F','var_jb_18_G','var_jb_18_O','queryreqformat_1|信贷审批查询',
                            'queryreason_02|贷款审批','queryreason_03|信用卡审批','queryreason_08|担保资格审查','state_3_1|正常','state_3_2|冻结','state_3_3|止付',
                            'state_3_4|销户','state_3_5|呆帐','state_3_6|未激活','guaranteetype_3_1|质押（含保证金）','guaranteetype_3_2|抵押','guaranteetype_3_3|保证',
                            'guaranteetype_3_4|信用/免担保','guaranteetype_3_6|组合（不含保证）','guaranteetype_3_9|其他','currency_3_人民币','currency_3_加拿大元','currency_3_日元',
                            'currency_3_欧元','currency_3_港元','currency_3_澳大利亚元','currency_3_澳门元','currency_3_瑞士法郎','currency_3_美元',
                            'currency_3_英镑','financetype_3_住房储蓄银行','financetype_3_商业银行','financetype_3_外资银行','var_jb_62_N','var_jb_62_Y',
                            'var_jb_57_N','var_jb_57_Y','var_jb_6_A','var_jb_6_L','var_jb_6_M','var_jb_6_O',
                            'var_jb_6_Q','var_jb_6_R','var_jb_6_S','state_4_1|正常','state_4_2|逾期','state_4_3|结清',
                            'state_4_4|呆帐','state_4_5|转出','paymentrating_01|日','paymentrating_02|周','paymentrating_03|月','paymentrating_04|季',
                            'paymentrating_05|半年','paymentrating_06|年','paymentrating_07|一次性','paymentrating_08|不定期','paymentrating_99|其他','guaranteetype_4_1|质押（含保证金）',
                            'guaranteetype_4_2|抵押','guaranteetype_4_3|保证','guaranteetype_4_4|信用/免担保','guaranteetype_4_5|组合（含保证）','guaranteetype_4_6|组合（不含保证）','guaranteetype_4_7|农户联保',
                            'guaranteetype_4_9|其他','currency_4_人民币','currency_4_港元','type_11|个人住房贷款','type_12|个人商用房（包括商住两用）贷款','type_13|个人住房公积金贷款',
                            'type_21|个人汽车贷款','type_31|个人助学贷款','type_41|个人经营性贷款','type_51|农户贷款','type_91|个人消费贷款','type_99|其他',
                            'financetype_4_住房储蓄银行','financetype_4_住房公积金管理中心','financetype_4_信托投资公司','financetype_4_商业银行','financetype_4_外资银行','financetype_4_小额信贷公司',
                            'financetype_4_机构','financetype_4_村镇银行','financetype_4_汽车金融公司','financetype_4_消费金融有限公司','financetype_4_财务公司','financetype_4_金融租赁公司',
                            'class5state_1|正常','class5state_2|关注','class5state_3|次级','class5state_4|可疑','class5state_5|损失','class5state_9|未知',
                            'var_jb_48_专升本','var_jb_48_专科','var_jb_48_专科(高职)','var_jb_48_博士研究生','var_jb_48_夜大电大函大普通班','var_jb_48_本科',
                            'var_jb_48_硕士研究生','var_jb_49_毕业','var_jb_49_结业','var_jb_55_*','var_jb_55_不详','var_jb_55_业余',
                            'var_jb_55_全日制','var_jb_55_函授','var_jb_55_在职','var_jb_55_夜大学','var_jb_55_开放教育','var_jb_55_普通全日制',
                            'var_jb_55_电视教育','var_jb_55_网络教育','var_jb_55_脱产','var_jb_55_非全日制','var_jb_47_*','var_jb_47_业余',
                            'var_jb_47_全日制','var_jb_47_函授','var_jb_47_夜大学','var_jb_47_开放教育','var_jb_47_成人','var_jb_47_普通',
                            'var_jb_47_普通全日制','var_jb_47_电视教育','var_jb_47_研究生','var_jb_47_网络教育','var_jb_47_脱产','var_jb_47_自学考试',
                            'var_jb_47_自考','var_jb_47_非全日制','var_jb_54_*','var_jb_54_业余','var_jb_54_全日制','var_jb_54_函授',
                            'var_jb_54_夜大学','var_jb_54_开放教育','var_jb_54_成人','var_jb_54_普通','var_jb_54_普通全日制','var_jb_54_电视教育',
                            'var_jb_54_研究生','var_jb_54_网络教育','var_jb_54_脱产','var_jb_54_自学考试','var_jb_54_自考','var_jb_54_非全日制',
                            'financetype_商业银行','currency_人民币','currency_日元','currency_美元','guaranteetype_1|质押（含保证金）','guaranteetype_2|抵押',
                            'guaranteetype_3|保证','guaranteetype_4|信用/免担保','guaranteetype_5|组合（含保证）','guaranteetype_6|组合（不含保证）','guaranteetype_9|其他','state_1|正常',
                            'state_2|冻结','state_3|止付','state_4|销户','state_5|呆帐','state_6|未激活','var_jb_39_B',
                            'var_jb_39_C','var_jb_39_D','var_jb_39_F','var_jb_39_K','var_jb_39_L','var_jb_39_N',
                            'var_jb_39_S','var_jb_39_U','var_jb_39_X','var_jb_39_Y','var_jb_93_M','var_jb_93_X',
                            'var_jb_93_Z','sub_divide']


def apply_log1p_transformation(dataframe, columns):
    for column in columns:
        if types.is_int64_dtype(dataframe[column]):
            dataframe[column] = np.log1p(dataframe[column])

    return dataframe
