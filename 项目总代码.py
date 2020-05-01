# -*- coding: utf-8 -*-
"""
Created on Thur Apr 30th 23:08:09 2020

@author: Dicast
"""
import os
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt  
import math

### 提取68路的GPS数据以及数据预处理
# 设置当前工作路径
os.chdir('C:/Users/Administrator/Desktop/python深圳68路公交客流量分析')
# 读取数据以及数据预处理
data = pd.DataFrame()
file_list = os.listdir('gps') #设置需要读取文件的文件夹
for file_name in file_list:
    csv_data = pd.read_csv('gps/'+file_name,delimiter=',',encoding='gbk') #对从9日到13日逐个文件读取
    csv_data.dropna() #去除任何有空值的行
    csv_data.drop_duplicates(['业务时间', '卡片记录编码', '车牌号'], keep='first', inplace=True)  # 去除在'业务时间','卡片记录编码','车牌号'上重复的行
    data = pd.concat([data,csv_data],axis=0) #合并所有文件数据
print('所有线路刷卡记录数据预处理后的数据形状：',data.shape) #查看预处理后的数据形状
#在预处理后的总GPS数据中取出68路的数据
data_68 = data.loc[(data['线路名称']=='68路'),:]
print('68路的刷卡记录数据为：',data_68)
print('68路的刷卡记录数据形状：',data_68.shape)

### 绘制68路刷卡位置的散点图
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['font.sans-serif']=['SimHei'] #这行代码用于显示中文，'SimHei'就是黑体
plt.figure(figsize=(8,4)) #画布大小
plt.scatter(data_68["经度"],data_68["纬度"])
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('68路的上车位置平面图')
plt.savefig('68路的上车位置平面图.png')
plt.show()

### 绘制所有公交线路的时刻-上车人数折线图
file_list = os.listdir('time') #设置需要读取文件的文件夹
plt.figure(figsize=(8,4)) #画布大小
for file_name in file_list:
    data_time = pd.read_csv('time/'+file_name) #对从9日到13日逐个文件读取
    plt.plot(data_time['date'],data_time['num'],label=file_name) #以一日内的时刻date为x轴，以上车人数num为y轴
plt.xticks([5,10,15,20],['5:00','10:00','15:00','20:00']) #将时刻改为时间
plt.xlabel('时间点')
plt.yticks([0,100000,200000,300000 ,400000,500000,600000],['0','10','20','30','40','50','60']) #将人数改为以“万”单位
plt.ylabel('刷卡人数（单位：万）')
plt.title('所有公交线路的时刻-上车人数折线图')
plt.legend(("2014-06-09","2014-06-10","2014-06-11","2014-06-12","2014-06-13")) #图例
plt.savefig('所有公交线路的时刻-上车人数折线图.png')
plt.show()

###密度聚类
data_68.reset_index(drop=True,inplace = True) #由于后面需要横向合并，重新创建序列
db = DBSCAN(eps=0.0011,min_samples=3).fit(data_68.iloc[:,0:2]) # DBSCAN聚类，半径为0.0011（度），3代表聚类中心的区域必须至少有3个才能聚成一类
flag = pd.Series(db.labels_, name=('flag')) #db.labels_是聚类数据的簇标签,labels_为-1为噪声点
df_cluster_result = pd.concat([data_68, flag], axis=1) #横向合并簇标签和原数据
df_cluster_result = df_cluster_result[df_cluster_result["flag"] >= 0] #去掉噪声点
print('68路的刷卡记录数据及其所在核心点flag为：',df_cluster_result) #flag值表示该乘客上车时所在站点的聚类核心点，一共有从0到40共41个核心点，将该flag值看作实际乘客上车站点。
df_cluster_result.to_csv('68_cluster_result.csv')
#绘制密度聚类后的68路核心上车点的散点图
df_station = df_cluster_result.copy()
df_station = df_station.drop_duplicates("flag")  # 去除重复值
df_station = df_station.reset_index(drop=False,inplace=False) #创建一个新序列，默认为drop=False,inplace=False
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['font.sans-serif']=['SimHei'] #这行代码用于显示中文，'SimHei'就是黑体
plt.scatter(df_station["经度"],df_station["纬度"],c=df_station["flag"])
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('68路的上车位置DBSCAN聚类核心点平面图')
plt.savefig('68路的上车位置DBSCAN聚类核心点平面图.png')
plt.show()

###计算每个站点的上车人数，将df_cluster_result中的flag值看作实际乘客上车站点
get_on_num = df_cluster_result.groupby('flag').count().iloc[:,1] #以flag值分组并计算每组记录的个数，这就是该flag的上车人数
get_on_num.name = ('get_on_num') #将列名改为get_on_num
get_on_num = get_on_num.reset_index()
print('各个站点的上车人数为：',get_on_num)

###下面我们尝试估计出每个站点的下车人数
#计算每个站点对乘客的吸引权重（吸引权重=该站上车人数/总上车人数）。由于多人上车的点自然也会多人下车，所以吸引权重越大，意味着乘客下车的概率越大。
attraction_weight = get_on_num['get_on_num']/sum(get_on_num['get_on_num'])
attraction_weight = pd.DataFrame({'weight':attraction_weight.values,'flag':attraction_weight.index}) #将attraction_weight由series类型转换为dataframe类型
print('各个站点的吸引权重为：',attraction_weight)
attraction_weight.to_csv('attraction_weight.csv')

#构建下车概率矩阵，i行j列表示从i站上车的乘客在j站下车的概率
#当不受权重影响时，自乘客上车起，乘客经过几站下车服从泊松分布，分布均值为，68路公交出行途经的站点数的数学期望
flag_num = len(attraction_weight['flag']) #站点数
lam = sum(attraction_weight['flag'])/flag_num #68路公交出行途经的站点数的数学期望
get_off_probability = pd.DataFrame(np.zeros((flag_num,flag_num+1))) #构建下车概率矩阵，i行j列表示从i站上车的乘客在j站下车的泊松部分概率.第flag_num+1列将为后面for循环计算i站上车乘客在所有站下车概率的总和起作用。
for i in range(flag_num):
    for j in range(flag_num):
        if(i<j): #i<j意味上车站点一定在下车站点之前
            get_off_probability.iloc[i,j] = (math.e**(-lam)*lam**(j-i))/math.factorial(j-i) #从i站上车的乘客在j站下车的泊松部分
    get_off_probability.iloc[i,flag_num] = sum(get_off_probability.iloc[i,0:flag_num]*attraction_weight.iloc[:,0]) #第flag_num+1列为从i站上车到所有站点的泊松概率乘以吸引权重的总和
#下车概率矩阵补上吸引权重部分
for i in range(flag_num):
    for j in range(flag_num):
        if(get_off_probability.iloc[i,flag_num]!=0):
            get_off_probability.iloc[i, j] =(get_off_probability.iloc[i, j] * attraction_weight.iloc[j, 0])/get_off_probability.iloc[i,flag_num] #加上吸引权重后，i行j列表示从i站上车的乘客在j站下车的概率
print('下车概率矩阵：',get_off_probability)
get_off_probability.to_csv('get_off_probability.csv')

#构建OD矩阵,求出一个站点到另一个站点的下车人数。OD矩阵i行j列表示从i站上车到j站下车的人数,最后一列是该站上车总人数，最后一行是该站下车总人数
OD = pd.DataFrame(np.zeros((flag_num+1,flag_num+1)))
for i in range(flag_num):
    for j in range(flag_num):
        if(i<j):
            OD.iloc[i,j] = round(get_on_num.iloc[i,1]*get_off_probability.iloc[i,j]) #从i站上车到j站下车的人数=从i站上车人数*从i站到j站的下车概率
for i in range(flag_num):
    OD.iloc[i,flag_num] = sum(OD.iloc[i,0:flag_num]) #从i站上车的总人数
for j in range(flag_num):
    OD.iloc[flag_num,j] = sum(OD.iloc[0:flag_num,j]) #从j站下车的总人数
OD.iloc[flag_num,flag_num] = sum(OD.iloc[flag_num,0:flag_num]) #总乘客数
print('OD矩阵为：',OD)
OD.to_csv('68_OD.csv')

###绘制各个站点的上下车人数柱状图
#创建一个dataframe，将每个站点的估计下车人数和实际上车人数放一起
get_off_num = []
for j in range(flag_num):
    get_off_num.append(int(OD.iloc[flag_num,j]))
get_off_num = pd.DataFrame(get_off_num,columns=['get_off_num'])
sum_get_num = pd.concat([get_on_num['get_on_num'], get_off_num], axis=1) #将下车人数与之前的上车人数get_on_num横向合并
print('各个站点的上车人数和下车人数为：',sum_get_num)
sum_get_num.to_csv('sum_get_num.csv')
#创建柱状图
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['font.sans-serif']=['SimHei'] #这行代码用于显示中文，'SimHei'就是黑体
sum_get_num.plot(kind='bar',figsize=(12,4))
plt.legend(['上车人数','下车人数'])
plt.xlabel('站点')
plt.ylabel('人数')
plt.title('68路各个站点的上车人数和下车人数')
plt.savefig('68路各个站点的上车人数和下车人数.png')
plt.show()
