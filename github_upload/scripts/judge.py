import json
import pandas as pd
import numpy as np
import function_v2 as F2
import functions as F
import functions_V3 as F3
import re

#提取数字
def matchnum(str1):
    if isinstance(str1, str):
        s = str1.strip()
        if s == '':
            return np.nan
        # 提取第一个数字（支持负号与小数），避免拼接多个数字导致解析失败
        m = re.search(r'-?\d+(?:\.\d+)?', s)
        return float(m.group(0)) if m else np.nan
    else:
        # 非字符串：如果是 NaN，按 NaN 返回；否则原样返回（数值可直接比较）
        try:
            if pd.isna(str1):
                return np.nan
        except Exception:
            pass
        return str1


#判断管径是否对称,输入为某血管管径dataframe以及阈值， 阈值2小于阈值1（阈值2可为none）
def judge_pipe(dataframe, threshold1, threshold2):
    ans = ''
    indexlist = dataframe.index
    LRlist = ['右','左']
    #ans = ans + F3.writeV_2(dataframe)
    
    for i in range(len(indexlist)):
        #根据有无残余管径判断是否对称
        if '管径' in indexlist[i] and '残余管径' not in indexlist[i]:
            tempseries = dataframe.loc[indexlist[i]]
            tempseries2 = dataframe.loc[indexlist[i].replace('管径', '残余管径' )]
            if pd.isna(tempseries2.iloc[0]):
                if not pd.isna(tempseries2.iloc[1]):
                    ans = ans + '双侧' + indexlist[i] + '不对称。'
            else:
                if pd.isna(tempseries2.iloc[1]):
                    ans = ans + '双侧' + indexlist[i] + '不对称。'
            
            
            ans = ans + F3.writeV_21(tempseries)
            ans = ans + F3.writeV_21(tempseries2)
            
            for j in range(2):
                if not pd.isna(tempseries.iloc[j]):
                    if not pd.isna(threshold2):
                        num = matchnum(tempseries.iloc[j])
                        if not pd.isna(num) and num < threshold2:
                            ans = ans + LRlist[j] + indexlist[i].replace('管径', '' ) + '先天发育不良。'
                            continue
                    
                    num = matchnum(str(tempseries.iloc[j]))
                    if not pd.isna(num) and num < threshold1:
                        ans = ans + LRlist[j] + indexlist[i].replace('管径', '' ) + '较细。'
           
                if not pd.isna(tempseries2.iloc[j]):
                    num2 = matchnum(tempseries2.iloc[j])
                    num1 = matchnum(tempseries.iloc[j])
                    if not pd.isna(num2) and not pd.isna(num1) and num1 != 0:
                        ans = ans + LRlist[j] + indexlist[i] + '剩余占比' + (f'{(num2/num1):.2f}') + '。'
                    
        elif '管径' not in indexlist[i] and '残余管径' not in indexlist[i]:
            ans = ans + F3.writeV_21(dataframe.loc[indexlist[i]])
            if '内膜' in indexlist[i]:
                tempseries = dataframe.loc[indexlist[i]]
                for j in range(2):
                    num = matchnum(tempseries.iloc[j])
                    if not pd.isna(num) and 1.5 > num > 1:
                        ans = ans + LRlist[j] + indexlist[i] + '增厚。'
                    elif not pd.isna(num) and num >= 1.5:
                        ans = ans + LRlist[j] + indexlist[i] + '严重增厚（存在斑块）。'
                        
    return ans

#椎动脉以及锁骨下动脉管径判断
def judge_pipe2(dataframe, threshold1, threshold2):
    ans = ''
    indexlist = dataframe.index
    LRlist = ['右','左']
    
    def repipe(thestr):
        templist = thestr.split('/')
        temprepipe = templist[0]
        temppipe = templist[1]
        return matchnum(temprepipe) / matchnum(temppipe)
    
    def jpipe(thestr, threshold1, threshold2):
        if not pd.isna(threshold2):
            num = matchnum(thestr)
            if not pd.isna(num) and num < threshold2:
              return '先天发育不良。'
        
        num = matchnum(thestr)
        if not pd.isna(num) and num < threshold1:
            return '较细。'
        else:
            return None
        
    
    for i in range(len(indexlist)):
        #根据有无残余管径判断是否对称
        if '管径' in indexlist[i] :
            tempseries = dataframe.loc[indexlist[i]]
            if not tempseries.iloc[0] or not tempseries.iloc[1]:
                if '/' not in tempseries.iloc[0] and '/' not in tempseries.iloc[1]:
                    ans = ans + F3.writeV_21(tempseries)
                    for j in range(2):
                        if not pd.isna(tempseries.iloc[j]):
                            jp = jpipe(tempseries.iloc[j], threshold1, threshold2)
                            if not pd.isna(jp):
                                ans = ans + LRlist[j] + indexlist[i] + jp
                elif '/' in tempseries.iloc[0] and '/' not in tempseries.iloc[1]:
                    ans = ans + '双侧' + indexlist[i] + '不对称。'
                    ans = ans + F3.writeV_21(tempseries)
                    ans = ans + '右' + indexlist[i] + '剩余占比' + (f'{repipe(tempseries.iloc[0]):.2f}') + '。'
                    tempstr = tempseries.iloc[0].split('/')[1]
                    jp = jpipe(tempstr, threshold1, threshold2)
                    if not pd.isna(jp):
                        ans = ans + '右' + '原始' + indexlist[i] + jp
                    jp2 = jpipe(tempseries.iloc[1], threshold1, threshold2)
                    if not pd.isna(jp2):
                        ans = ans + '左' + indexlist[i] + jp2
                elif '/' not in tempseries.iloc[0] and '/'  in tempseries.iloc[1]:
                    ans = ans + '双侧' + indexlist[i] + '不对称。'
                    ans = ans + F3.writeV_21(tempseries)
                    ans = ans + '右' + indexlist[i] + '剩余占比' + (f'{repipe(tempseries.iloc[0]):.2f}') + '。'
                    tempstr = tempseries.iloc[1].split('/')[1]
                    jp = jpipe(tempseries.iloc[0], threshold1, threshold2)
                    if not pd.isna(jp):
                        ans = ans + '右' + indexlist[i] + jp
                    jp2 = jpipe(tempstr, threshold1, threshold2)
                    if not pd.isna(jp2):
                        ans = ans + '左' + '原始' +indexlist[i] +  jp2
                else:
                    ans = ans + F3.writeV_21(tempseries)
                    for j in range(2):
                        ans = ans + LRlist[j] + indexlist[i] + '剩余占比' + (f'{repipe(tempseries.iloc[j]):.2f}') + '。'
                        tempstr = tempseries.iloc[j].split('/')[1]
                        jp = jpipe(tempstr, threshold1, threshold2)
                        if not pd.isna(jp):
                            ans = ans + LRlist[j] + '原始' + indexlist[i] + jp
    return ans
'''                        
        if '内膜' in indexlist[i]:
            tempseries = dataframe.loc[indexlist[i]]
            for j in range(2):
                if not pd.isna(tempseries[j]) and 1.5> matchnum(tempseries[j])> 1:
                    ans = ans + LRlist[j] + indexlist[i] + '增厚'
            '''            
#比大小
def judge(number, standard):
    for i in range(3):
        if number <= standard[i]:
            return i
        else:
            continue
    return 3

#
def judge_write(number):
    ans = ''
    if number == 0:
        ans = '正常'
    elif number == 1:
        ans = '轻微升高'
    elif number == 2:
        ans = '中度升高' 
    elif number == 3:
        ans = '严重升高'
    return ans 
    
    
#判断流速
def judge_v(series, standard, judgenumber, paranumber, lownumber=0):#series为各血管流速字典左或右series，judgenumber为控制输出程度， paranumber为计算比值参数值
    ans = ''
    indexlist = series.index
    for i in range(len(indexlist)):
        thedata = series[indexlist[i]]
        if not pd.isna(thedata):
            val = matchnum(thedata)
            if pd.isna(val):
                continue
            if '峰值' in indexlist[i]:
                ansnumber = judge(val, standard['PV'])
                if ansnumber >= judgenumber:
                    ans = ans + indexlist[i] + judge_write(ansnumber) + '。'
                    if not pd.isna(paranumber):
                        pval = matchnum(paranumber)
                        if not pd.isna(pval) and pval != 0:
                            temppara = val / pval
                            ansnumber = judge(temppara, standard['para'])
                            if ansnumber >= judgenumber:
                                ans = ans + indexlist[i] + '比值参数' + judge_write(ansnumber) + '。'
            elif '舒张' in indexlist[i]:
                ansnumber = judge(val, standard['DV'])
                if ansnumber >= judgenumber:
                    ans = ans + indexlist[i] + judge_write(ansnumber)+ '。'
            elif '平均流速' in indexlist[i]:
                ansnumber = judge(val, standard['MV'])
                if ansnumber >= judgenumber:
                    ans = ans + indexlist[i] + judge_write(ansnumber)+ '。'
                elif val < lownumber:
                    ans = ans + indexlist[i] + '降低。'
    
    if ans == '':
        ans = ans + '流速正常。'
    
    return ans


    

#判断椎动脉流速，输入为椎动脉流速字典和标准字典
def judge_va_v(vadict, standard):
    tempstandard = standard["椎动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        
        tempseries = vadict[LRlist[i]]
        paranumber = tempseries['椎动脉峰值流速']
        if  judge_v(tempseries, tempstandard, 1, paranumber) != '':
            ans = ans + LRlist[i] +'椎动脉'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 1, paranumber)
            ans = ans.replace('开口处', 'V1段')
    return ans
#判断锁骨下动脉流速，输入为椎动脉流速字典和标准字典
def judge_suba_v(vadict, standard):
    tempstandard = standard["锁骨下动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] +'锁骨下动脉'+ ':'
        tempseries = vadict[LRlist[i]]
        paranumber = tempseries['锁骨下动脉峰值流速']
        if  judge_v(tempseries, tempstandard, 3, paranumber) != '':
            ans = ans + LRlist[i] +'锁骨下动脉'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 3, paranumber)
            ans = ans.replace('锁骨下动脉', '')
    
    return ans

#判断颈内动脉流速， 输入为颈动脉流速字典和标准字典
def judge_ica_v(vadict, standard, paraseries):
    tempstandard = standard["颈内动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] +'颈内动脉'+ ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        if  judge_v(tempseries, tempstandard, 2, paranumber) != '':
            ans = ans + LRlist[i] +'颈内动脉'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 2, paranumber)
            ans = ans.replace('颈内动脉', '')
    
    return ans

#判断颈总动脉流速， 输入为颈动脉流速字典和标准字典
def judge_cca_v(vadict, standard, paraseries):
    tempstandard = standard["颈总动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] +'颈总动脉'+ ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        if  judge_v(tempseries, tempstandard, 2, paranumber) != '':
            ans = ans + LRlist[i] +'颈总动脉'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 2, paranumber)
            ans = ans.replace('颈总动脉', '')
    return ans
#判断颈外动脉流速， 输入为颈动脉流速字典和标准字典
def judge_eca_v(vadict, standard, paraseries):
    tempstandard = standard["颈外动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] +'颈外动脉'+ ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        if  judge_v(tempseries, tempstandard, 2, paranumber) != '':
            ans = ans + LRlist[i] +'颈外动脉'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 2, paranumber)
            ans = ans.replace('颈外动脉', '')
    return ans
#判断颈动脉球流速， 输入为颈动脉流速字典和标准字典
def judge_bulb_v(vadict, standard, paraseries):
    tempstandard = standard["颈动脉球"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] +'颈动脉球'+ ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        if  judge_v(tempseries, tempstandard, 2, paranumber) != '':
            ans = ans + LRlist[i] +'颈动脉球'+ ':'
            ans = ans + judge_v(tempseries, tempstandard, 2, paranumber)
            ans = ans.replace('颈动脉球', '')
    return ans

#以下为颅内动脉流速诊断
#判断大脑中动脉流速， 输入为流速字典和标准字典
def judge_mca_v(vadict, standard):
    tempstandard = standard["大脑中动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        #ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        #paranumber = paraseries[LRlist[i]]
        paranumber = tempseries['大脑中动脉远端1峰值流速']
        if  judge_v(tempseries, tempstandard, 1, paranumber, lownumber=43) != '':
            ans = ans + LRlist[i] + ':'
            ans = ans + judge_v(tempseries, tempstandard, 1, paranumber)
        ans = ans.replace('大脑中动脉', '')
    
    return ans
#大脑前动脉
def judge_aca_v(vadict, standard, paraseries):
    tempstandard = standard["大脑前动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber, lownumber=39)
        ans = ans.replace('大脑前动脉', '')  
    return ans
#大脑后动脉
def judge_pca_v(vadict, standard, paraseries):
    tempstandard = standard["大脑后动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber, lownumber=30)
        ans = ans.replace('大脑后动脉', '')  
    return ans
#椎动脉颅内段
def judge_vadist_v(vadict, standard):
    tempstandard = standard["椎动脉颅内段"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = tempseries['椎动脉颅内段远端峰值流速']
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber, lownumber=28)
        ans = ans.replace('椎动脉颅内段', '')  
    return ans
#TCCD颈内动脉
def judge_tccdica_v(vadict, standard):
    tempstandard = standard["TCCD颈内动脉"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = None
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber, lownumber=30)
        ans = ans.replace('TCCD颈内动脉', '')  
    return ans
#颈内动脉虹吸段
def judge_cs_v(vadict, standard):
    tempstandard = standard["颈内动脉虹吸段"]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = None
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber, lownumber=30)
        ans = ans.replace('颈内动脉虹吸段', '')  
    return ans
#基底动脉
def judge_ba_v(series, standard, paranumber):
    tempstandard = standard["基底动脉"]
    ans = ''
    ans = ans + judge_v(series, tempstandard, 1, paranumber, lownumber=31)
    ans = ans.replace('基底动脉', '')  
    ans = ans.replace('右左', '') 
    return ans

def judge_brain_v(thedict ,name, standard, paraseries):
    vadict = thedict[name]
    tempstandard = standard[name]
    ans = ''
    LRlist = ['右','左']
    for i in range(2):
        ans = ans + LRlist[i] + ':'
        tempseries = vadict[LRlist[i]]
        paranumber = paraseries[LRlist[i]]
        ans = ans + judge_v(tempseries, tempstandard, 1, paranumber)
        ans = ans.replace(name, '')
    
    return ans