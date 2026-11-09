import json
import pandas as pd
import numpy as np
import function_v2 as F2

#生成句子
def write(df):
    ans = ''
    for i in range(2):
        if not pd.isna(df['data'][i]):
            if '斑块' in df['LR'][i]:
                ans = ans + df['LR'][i] + ':' + str(df['data'][i]) + '。' #+ '\n'
            elif '管径' in df['LR'][i] or '内膜' in df['LR'][i]:
                ans = ans + df['LR'][i] + '为' + str(df['data'][i]) + 'mm。' #+ '\n'
            elif '流速' in df['LR'][i]:
                ans = ans + df['LR'][i] + '为' + str(df['data'][i]) + 'cm/s。' #+ '\n'
            elif '搏动指数' in df['LR'][i]:
                ans = ans + df['LR'][i] + '为' + str(df['data'][i]) + '。'
    return ans

#检测是否左右有数据
def checkLR(list, i):
    temp_index = list.index[i]
    temp_index = temp_index.replace('右', '左')
    if pd.isna(list[i]) and pd.isna(list[temp_index]):
        return False
    return True

#得到左侧数据
def getleftdata(list, i):
    temp_index = list.index[i]
    temp_index = temp_index.replace('右', '左')
    data = [[list.index[i], list[i]], [temp_index, list[temp_index]]]
    df = pd.DataFrame(data, columns=['LR', 'data'])
    return df

#写V0的数据，数据不重排
def weiteV0(series):
    ans = ''
    for i in range(6,376):
        if not pd.isna(series[i]):
            if '斑块' in series.index[i]:
                ans = ans + series.index[i] + ':' + str(series[i]) + '。' #+ '\n'
            elif '管径' in series.index[i] or '内膜' in series.index[i]:
                ans = ans +series.index[i] + '为' + str(series[i]) + 'mm。'# + '\n'
            elif '流速' in series.index[i]:
                ans = ans + series.index[i] + '为' + str(series[i]) + 'cm/s。' #+ '\n'
            elif '搏动指数' in series.index[i]:
                ans = ans + series.index[i] + '为' + str(series[i]) + '。'
    return ans

#写V1版本的数据，只有颈部数据重排
def weiteV1(series):
    ans = ''
    for i in range(6, 188):
        if '右' in series.index[i]:
            if checkLR(series, i):
                temp_df = getleftdata(series, i)
                temp_ans = write(temp_df)
                ans = ans + temp_ans
    if not pd.isna(series.iloc[188]):            
        ans = ans + series.iloc[188]
    for i in range(189, 376):
        if '右' in series.index[i]:
            if checkLR(series, i):
                temp_df = getleftdata(series, i)
                temp_ans = write(temp_df)
                ans = ans + temp_ans
    if not pd.isna(series.iloc[377]):            
        ans = ans + series.iloc[377]
    return ans

#V1.1数据，所有部分重排
def weiteV1_1(series):
    ans = ''
    for i in range(6, 188):
        if '右' in series.index[i]:
            if checkLR(series, i):
                temp_df = getleftdata(series, i)
                temp_ans = write(temp_df)
                ans = ans + temp_ans
    if not pd.isna(series.iloc[188]):            
        ans = ans + series.iloc[188]
    brainbv = F2.classifybv_b(series)
    for i in range(8):
        tempbv = brainbv.iloc[i]
        for j in range(len(tempbv)):
            if '右' in tempbv.index[j]:
                if checkLR(tempbv, j):
                    temp_df = getleftdata(tempbv, j)
                    temp_ans = write(temp_df)
                    ans = ans + temp_ans
    if not pd.isna(series.iloc[377]):            
        ans = ans + series.iloc[377]
    return ans
        
        
        
        
        
        
        
        
        
        
        
        

