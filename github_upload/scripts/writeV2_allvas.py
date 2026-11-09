import json
import pandas as pd
import numpy as np
import functions as F
import function_v2 as F2
import functions_V3 as F3
import judge as J

#写V2颈内动脉数据
def write_V2_ica(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + J.judge_pipe(thedict['管径'], 3.5, None)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_ica_v(thedict['流速'], standard, paraseries)
    return ans

#写V2颈总动脉数据
def write_V2_cca(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + J.judge_pipe(thedict['管径'], 6.0, None)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_cca_v(thedict['流速'], standard, paraseries)
    return ans
#写V2颈外动脉数据
def write_V2_eca(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + J.judge_pipe(thedict['管径'], 3.5, None)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_eca_v(thedict['流速'], standard, paraseries)
    return ans
#写V2颈动脉球数据
def write_V2_bulb(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + J.judge_pipe(thedict['管径'], 3.5, None)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_bulb_v(thedict['流速'], standard, paraseries)
    return ans

#写V2椎动脉数据
def write_V2_va(thedict, standard):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + F3.writeV_2(thedict['管径'])
    ans = ans + J.judge_pipe2(thedict['管径'], 2.5, 2.0)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_va_v(thedict['流速'], standard)
    return ans
#写V2锁骨下动脉数据
def write_V2_suba(thedict, standard):
    ans = F3.writeV_2(thedict['斑块'])
    ans = ans + F3.writeV_2(thedict['管径'])
    ans = ans + J.judge_pipe2(thedict['管径'], 2.5, None)
    ans = ans + F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_suba_v(thedict['流速'], standard)
    return ans

#写V2颈部所有数据
def write_V2_neck(thedict, standard, oridata):
    ans = ''
    tempparaseries = thedict['颈内动脉']['流速'].loc['颈内动脉远端峰值流速']
    ans = ans + write_V2_cca(thedict['颈总动脉'], standard, tempparaseries)
    ans = ans+ ' \n'
    ans = ans + write_V2_bulb(thedict['颈动脉球'], standard, tempparaseries)
    ans = ans+ ' \n'
    ans = ans + write_V2_ica(thedict['颈内动脉'], standard, tempparaseries)
    ans = ans+ ' \n'
    ans = ans + write_V2_eca(thedict['颈外动脉'], standard, tempparaseries)
    ans = ans+ ' \n'
    ans = ans + write_V2_va(thedict['椎动脉'], standard)
    ans = ans+ ' \n'
    if not pd.isna(oridata['走行变异右侧']):
        ans = ans + '右侧椎动脉走行变异:' +  oridata['走行变异右侧']
    if not pd.isna(oridata['走行变异左侧']):
        ans = ans + '左侧椎动脉走行变异:' +  oridata['走行变异左侧']
    ans = ans + write_V2_suba(thedict['锁骨下动脉'], standard)
    ans = ans+ ' \n'
    return ans

#写V2大脑中动脉数据
def write_V2_mca(thedict, standard):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_mca_v(thedict['流速'], standard)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2大脑前动脉数据
def write_V2_aca(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_aca_v(thedict['流速'], standard, paraseries)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2大脑后动脉数据
def write_V2_pca(thedict, standard, paraseries):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_pca_v(thedict['流速'], standard, paraseries)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2椎动脉颅内段数据
def write_V2_vadist(thedict, standard):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_vadist_v(thedict['流速'], standard)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2TCCD颈内动脉数据
def write_V2_tccdica(thedict, standard):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_tccdica_v(thedict['流速'], standard)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2颈内动脉虹吸段数据
def write_V2_cs(thedict, standard):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + J.judge_cs_v(thedict['流速'], standard)
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans
#写V2基底动脉数据
def write_V2_ba(thedict, standard, paraseries):
    ans = F3.writeV_2_ba(thedict['流速']['右左'])
    ans = ans + J.judge_ba_v(thedict['流速']['右左'], standard, paraseries)
    ans = ans + F3.writeV_2_ba(thedict['搏动与频谱']['右左'])
    return ans
#写V2眼动脉数据
def write_V2_oa(thedict):
    ans = F3.writeV_2(thedict['流速'])
    ans = ans + F3.writeV_2(thedict['搏动与频谱'])
    return ans

def write_V2_brain(thedict, standard):
    ans = write_V2_mca(thedict['大脑中动脉'], standard)
    ans = ans+ ' \n'
    ans = ans + write_V2_tccdica(thedict['TCCD颈内动脉'], standard)
    ans = ans.replace('TCCD', '')
    ans = ans+ ' \n'
    ans = ans + write_V2_cs(thedict['颈内动脉虹吸段'], standard)
    ans = ans+ ' \n'
    ans = ans + write_V2_aca(thedict['大脑前动脉'], standard, thedict['大脑中动脉']['流速'].loc['大脑中动脉远端1峰值流速'])
    ans = ans+ ' \n'
    ans = ans + write_V2_pca(thedict['大脑后动脉'], standard, thedict['大脑后动脉']['流速'].loc['大脑后动脉P2段峰值流速'])
    ans = ans+ ' \n'
    ans = ans + write_V2_vadist(thedict['椎动脉颅内段'], standard)
    ans = ans+ ' \n'
    ans = ans + write_V2_ba(thedict['基底动脉'], standard, thedict['椎动脉颅内段']['流速'].loc['椎动脉颅内段峰值流速']['右'])
    ans = ans+ ' \n'
    ans = ans + write_V2_oa(thedict['眼动脉'])
    return ans

def simplewrite(name, data):
    ans = ''
    if not pd.isna(data[name]):
        ans = ans + name +':'+ str(data[name]) + '。'
    return ans
        
def writeV2_all(standard, oridata):
    alldict= F3.getre_data(oridata)
    alldict2= F3.getbrainre_data(oridata)
    ans = ''
    ans = ans + write_V2_neck(alldict, standard, oridata)
    ans = ans + simplewrite('神内备注1', oridata)
    ans = ans.replace('神内备注1', '')
    ans = ans + write_V2_brain(alldict2, standard)
    ans = ans + simplewrite('侧支开放情况右侧', oridata)
    ans = ans + simplewrite('侧支开放情况左侧', oridata)
    ans = ans + simplewrite('神内备注2', oridata)
    ans = ans.replace('神内备注2', '')
    ans = ans.replace('右:左:', '')
    return ans