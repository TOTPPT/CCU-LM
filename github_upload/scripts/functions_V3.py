import json
import pandas as pd
import numpy as np
import function_v2 as F2
import functions as F

#V2级重构颈部数据
def getre_data(series):#输入为原始数据序列
    bvnames = ['颈总动脉', '颈动脉球', '颈内动脉', '颈外动脉', '椎动脉', '锁骨下动脉']
    re_data = {
        '颈总动脉':{}, '颈动脉球':{}, '颈内动脉':{}, '颈外动脉':{}, '椎动脉':{}, '锁骨下动脉':{}
        }
    bv_series = F2.classifybv(series)
    brainbv_series = F2. classifybv_b(series)
    for i in range(6):
        temp_data = bv_series[bvnames[i]]
        re_data[bvnames[i]] = F2.reconstractbv(temp_data)
    return re_data

#重构脑部每根血管数据, 输入为每根脑部血管数据
def reconstractbrainbv(series):
    # 预创建列为 object，避免首次写入数值导致列推断为 float64
    bdf = pd.DataFrame({'右': pd.Series(dtype='object'), '左': pd.Series(dtype='object'), '右左': pd.Series(dtype='object')})
    ldf = pd.DataFrame({'右': pd.Series(dtype='object'), '左': pd.Series(dtype='object'), '右左': pd.Series(dtype='object')})
    names = [ '流速', '搏动指数', '异常频谱', '低搏动性']
    for i in range(len(series)):
        for j in range(1,4):
            if names[j] in series.index[i]:
                if '右' in series.index[i]:
                    temp_index = series.index[i].replace('右侧', '')
                    bdf.loc[temp_index, '右'] = series.iloc[i]
                elif '左' in series.index[i]:
                    temp_index = series.index[i].replace('左侧', '')
                    bdf.loc[temp_index, '左'] = series.iloc[i]
                else:
                    temp_index = series.index[i]
                    bdf.loc[temp_index, '右左'] = series.iloc[i]
        if names[0] in series.index[i]:
            if '右' in series.index[i]:
                temp_index = series.index[i].replace('右侧', '')
                ldf.loc[temp_index, '右'] = series.iloc[i]
            elif '左' in series.index[i]:
                temp_index = series.index[i].replace('左侧', '')
                ldf.loc[temp_index, '左'] = series.iloc[i]
            else:
                temp_index = series.index[i]
                ldf.loc[temp_index, '右左'] = series.iloc[i]
    ans = {
            '流速':ldf, '搏动与频谱':bdf, 
            }
    return ans
 
#V2级重构脑部所有数据
def getbrainre_data(series):#输入为原始数据序列
    brainbvnames = ['大脑中动脉', 'TCCD颈内动脉', '大脑前动脉', '大脑后动脉', '颈内动脉虹吸段', '眼动脉', '椎动脉颅内段', '基底动脉']
    re_data_brain = {
        '大脑中动脉':{}, 'TCCD颈内动脉':{}, '大脑前动脉':{}, '大脑后动脉':{}, '颈内动脉虹吸段':{}, '眼动脉':{}, '椎动脉颅内段':{}, '基底动脉':{}
        }
    brainbv_series = F2. classifybv_b(series)
    for i in range(8):
        temp_data = brainbv_series[brainbvnames[i]]
        re_data_brain[brainbvnames[i]] = reconstractbrainbv(temp_data)
    return re_data_brain


#判断管径是否对称, 输入为血管管径字典(实际为DF) 阈值为0.3
def judge_symmetry(thedict):
    theindex = thedict.index
    ans = ''
    for i in range(len(theindex)):
        if '管径' in theindex[i]:
            temp = thedict.loc[theindex[i]]
            if not pd.isna(temp.iloc[0]) and not pd.isna(temp.iloc[1]):
                temp_ans = abs(float(temp.iloc[0]) - float(temp.iloc[1]))/max(float(temp.iloc[0]), float(temp.iloc[1]))
                if temp_ans > 0.3:
                    ans = ans + theindex[i] + '不对称。'
    return ans

#判断峰值流速 输入为血管名, 颈部或脑部字典, 标准字典
def judge_pv(name, bvdict, standard):
    ans = ''
    data = bvdict[name]['流速']
    thestandard = standard[name]['PV']
    tempname = name + '峰值流速'
    tempdata = data.loc[tempname]
    for i in range(2):
        if tempdata.iloc[i] < thestandard[0]:
            tempans = tempdata.index[i] + tempname + '流速正常。'
            ans += tempans
        elif tempdata.iloc[i] < thestandard[1]:
            tempans = tempdata.index[i] + tempname + '峰值流速轻度升高。' + tempdata.index[i] + tempname + '轻度狭窄'
            ans += tempans
        elif tempdata.iloc[i] < thestandard[2]:
            tempans = tempdata.index[i] + tempname + '峰值流速升高。' + tempdata.index[i] + tempname + '中度狭窄'
            ans += tempans 
        else:
            tempans = tempdata.index[i] + tempname + '峰值流速严重升高。' + tempdata.index[i] + tempname + '严重狭窄'
            ans += tempans
    return ans


#生成文本V2，输入为某根血管的某个具体字典，比如流速
def writeV_2(df):
    ans = ''
    for i in range(len(df)):
        tempseries = df.loc[df.index[i]]
        for j in range(2):
            if not pd.isna(tempseries.iloc[j]):
                if '斑块' in df.index[i] or '异常频谱' in df.index[i] or '低搏动性' in df.index[i]:
                    ans = ans + tempseries.index[j] + df.index[i] + ':' + str(tempseries.iloc[j]) + '。' 
                elif '管径' in df.index[i] or '内膜' in df.index[i]:
                    ans = ans + tempseries.index[j] +  df.index[i] + '为' + str(tempseries.iloc[j]) + 'mm。' 
                elif '流速' in df.index[i]:
                    ans = ans + tempseries.index[j] + df.index[i] + '为' + str(tempseries.iloc[j]) + 'cm/s。'
                elif '搏动指数' in df.index[i]:
                    ans = ans + tempseries.index[j] +  df.index[i] + '为' + str(tempseries.iloc[j]) + '。'
    
    return ans

def writeV_2_ba(tempseries):
    ans = ''
    for i in range(len(tempseries)):
        if not pd.isna(tempseries.iloc[i]):
            if '斑块' in tempseries.index[i] or '异常频谱' in tempseries.index[i] or '低搏动性' in tempseries.index[i]:
                ans = ans + tempseries.index[i]  + ':' + str(tempseries.iloc[i]) + '。' 
            elif '管径' in tempseries.index[i] or '内膜' in tempseries.index[i]:
                ans = ans + tempseries.index[i] + '为' + str(tempseries.iloc[i]) + 'mm。' 
            elif '流速' in tempseries.index[i]:
                ans = ans + tempseries.index[i]  + '为' + str(tempseries.iloc[i]) + 'cm/s。'
            elif '搏动指数' in tempseries.index[i]:
                ans = ans + tempseries.index[i]  + '为' + str(tempseries.iloc[i]) + '。'
    
    return ans


#生成文本V2.1，输入为某根血管的某个具体某行数据
def writeV_21(series):
    ans = ''
    LRlist = ['右','左']
    for j in range(2):
        if not pd.isna(series.iloc[j]):
            if '斑块' in series.name or '异常频谱' in series.name or '低搏动性' in series.name:
                ans = ans + LRlist[j] + series.name + ':' + str(series.iloc[j]) + '。' 
            elif '管径' in series.name or '内膜' in series.name:
                ans = ans + LRlist[j] +  series.name + '为' + str(series.iloc[j]) + 'mm。' 
            elif '流速' in series.name:
                ans = ans + LRlist[j] + series.name + '为' + str(series.iloc[j]) + 'cm/s。'
            elif '搏动指数' in series.name:
                ans = ans + LRlist[j] +  series.name + '为' + str(series.iloc[j]) + '。'
    
    return ans