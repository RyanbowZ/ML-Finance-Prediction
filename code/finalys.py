import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
import math

ts.set_token(config.token)
#本代码用于从tushare上获取10只具有典型代表性的股票日线数据

#获取正态分布曲线
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
def predicte(s,u,r,dt):
    return math.exp(math.log(s)+(u-0.5*r**2)*dt)
#将股票代码存至列表，按列表中元素进行依次处理
stock=['601318','601988','601939','600519','002230','000858','002594','601099','600019','600036']
for i in stock:
    if int(i)>600000:#区分上证和深证指数
        tscode=str(i)+'.SH'
    else:
        tscode=str(i)+'.SZ'
    df = ts.pro_bar(ts_code=tscode, adj='qfq', start_date='20160101', end_date='20210604')#读取该时间区间的股票数据
    df = df.sort_values(by="trade_date",ascending=True)#按时间升序对原始数据进行排序
    filepath="D:/Documents/finance/"+str(i)
    df.to_csv(filepath+'.csv')#转为.csv表格，便于读取分析


    ds = pd.read_csv(filepath+".csv",index_col="ts_code")[["pct_chg","close"]]#按特定项读入数据
    mean=ds['pct_chg'].mean()#求取收益率的平均值
    std=ds['pct_chg'].std()#求取收益率的方差
    maxd=int(max(-ds['pct_chg'].min(),ds['pct_chg'].max()))#求取收益率的极值的绝对值
    x = np.arange(-maxd-2, maxd+2,maxd/30)#设定绘图自变量区间，使可视化效果更佳
    y = normfun(x, mean, std)*len(ds)#拟合正态曲线，同时逆均一化
    plt.plot(x, y)#绘制正态曲线图
    n, bins, patches = plt.hist(ds['pct_chg'], bins=int(13+maxd), facecolor='green', alpha=0.75)#绘制频率直方图
    plt.xlabel('pct_chg')
    plt.title('Stock:'+i+'\'s distribution map')
    plt.text(-maxd-2,n.max(),"mean="+str(mean),fontsize=8)
    plt.text(-maxd-2,int(n.max()*29/30),"std="+str(std),fontsize=8)
    plt.text(-maxd-2,int(n.max()*28/30),"max="+str(ds['pct_chg'].max()),fontsize=8)
    plt.text(-maxd-2,int(n.max()*27/30),"mix="+str(ds['pct_chg'].min()),fontsize=8)
    plt.savefig(filepath+".png")
    plt.close()
