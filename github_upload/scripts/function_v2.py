import json
import pandas as pd
import numpy as np

#归类颈动脉各个血管的数据
def classifybv(series):
    CCA, BULB, ICA, ECA , VA , SUBA = pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series()
    bvlist = [CCA, BULB, ICA, ECA, VA, SUBA]
    bvnames = ['颈总动脉', '颈动脉球', '颈内动脉', '颈外动脉', '椎动脉', '锁骨下动脉']
    for i in range(6,188):
        for j in range(len(bvnames)):
            if bvnames[j] in series.index[i]:
                bvlist[j][series.index[i]] = series.iloc[i]
                break
    bvlist = pd.Series(bvlist, index=bvnames)
    return bvlist

#归类脑动脉各个血管的数据
def classifybv_b(series):
    MCA, ICA, ACA, PCA, CS, OA, VA, BA = pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series()
    bvlist = [MCA, ICA, ACA, PCA, CS, OA, VA, BA]
    bvnames = ['大脑中动脉', 'TCCD颈内动脉', '大脑前动脉', '大脑后动脉', '颈内动脉虹吸段', '眼动脉', '椎动脉颅内段', '基底动脉']
    for i in range(215,378):
        for j in range(len(bvnames)):
            if bvnames[j] in series.index[i]:
                bvlist[j][series.index[i]] = series.iloc[i]
                break
    bvlist = pd.Series(bvlist, index=bvnames)
    #bvlist = bvlist.rename(index={'颈内动脉':'TCCD颈内动脉'})
    return bvlist

#重构每个颈部血管的数据
def reconstractbv(series):
    # 预创建列为 object，避免首次写入数值导致列推断为 float64
    gdf = pd.DataFrame({'右': pd.Series(dtype='object'), '左': pd.Series(dtype='object')})
    bdf = pd.DataFrame({'右': pd.Series(dtype='object'), '左': pd.Series(dtype='object')})
    ldf = pd.DataFrame({'右': pd.Series(dtype='object'), '左': pd.Series(dtype='object')})
    names = ['管径', '内膜', '狭窄', '斑块', '流速']
    for i in range(len(series)):
        for j in range(3):
            if names[j] in series.index[i]:
                if '右' in series.index[i]:
                    temp_index = series.index[i].replace('右', '')
                    gdf.loc[temp_index, '右'] = series.iloc[i]
                elif '左' in series.index[i]:
                    temp_index = series.index[i].replace('左', '')
                    gdf.loc[temp_index, '左'] = series.iloc[i]
        if names[3] in series.index[i]:
            if '右' in series.index[i]:
                temp_index = series.index[i].replace('右', '')
                bdf.loc[temp_index, '右'] = series.iloc[i]
            elif '左' in series.index[i]:
                temp_index = series.index[i].replace('左', '')
                bdf.loc[temp_index, '左'] = series.iloc[i]
        if names[4] in series.index[i]:
            if '右' in series.index[i]:
                temp_index = series.index[i].replace('右', '')
                ldf.loc[temp_index, '右'] = series.iloc[i]
            elif '左' in series.index[i]:
                temp_index = series.index[i].replace('左', '')
                ldf.loc[temp_index, '左'] = series.iloc[i]
    ans = {
            '管径':gdf, '斑块':bdf, '流速':ldf
            }
    return ans

'''
#重新整理各个血管数据
def getre_data(series):
    bvnames = ['颈总动脉', '颈动脉球', '颈内动脉', '颈外动脉', '椎动脉', '锁骨下动脉']
    re_data = {
        '颈总动脉':{}, '颈动脉球':{}, '颈内动脉':{}, '颈外动脉':{}, '椎动脉':{}, '锁骨下动脉':{}
        }
    for i in range(6):
        temp_data = series[bvnames[i]]
        re_data[bvnames[i]] = reconstractbv(temp_data)
        
    return re_data
'''