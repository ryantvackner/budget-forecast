# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:02:26 2023

@author: rvackner
"""
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

cnxncis = pyodbc.connect('DSN=XXXXX;PWD=XXXXX')
cnxnsql = pyodbc.connect('DRIVER=XXXXX;SERVER=XXXXX;DATABASE=XXXXX;Trusted_Connection=XXXXX')
df_rdg = pd.read_sql_query("SELECT BI_ACCT, BI_BILL_DT_TM, BI_REV_YRMO, BI_USAGE, BI_DMD_RDG, BI_RATE_SCHED, BI_DMD_MULT, BI_MTR_NBR FROM XXXXX.XXXXX WHERE (BI_REV_YRMO >= 202111) AND (BI_REV_YRMO < 202311)", cnxncis)
df_rdg['BI_DMD_RDG'] = df_rdg['BI_DMD_RDG']*df_rdg['BI_DMD_MULT']
df_hist = pd.read_sql_query("SELECT BI_ACCT, MAX(BI_REV_CLASS_CD) AS BI_REV_CLASS_CD FROM XXXXX.XXXXX WHERE (BI_REV_YRMO >= 202111) AND (BI_REV_YRMO < 202311) GROUP BY BI_ACCT", cnxncis)
df = df_rdg.merge(df_hist, how='left', on='BI_ACCT')

"""
Import Charge CD Info
"""
rev_1_chg_cd = [71, 73, 81, 91, 101, 101.001, 103, 103.001, 104, 105.001, 107, 107.001,
                109, 109.001, 111.001, 113, 113.001, 114, 115.001, 121.001, 125.001,
                129, 130, 131, 132, 133, 136, 137, 138, 139, 141, 143, 144, 145, 147,
                156, 157, 158, 159, 189, 191]

rev_2_chg_cd = [71, 73, 81, 91, 101, 101.1, 105.1, 129, 130, 131, 133, 159]

rev_4_chg_cd = [11, 39, 71, 73, 81, 91, 101, 101.001, 103, 103.001, 104, 105, 105.001, 106, 107, 107.001,
                108, 109, 109.001, 110, 110.001, 112, 113, 113.001, 115, 115.001, 117, 118, 119.001,
                120, 121, 121.001, 122, 122.001, 125.001, 127, 129, 130, 131, 132, 133, 134, 135,
                136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 151, 152, 153, 155, 156, 157,
                158, 159, 164, 182, 183, 185, 189, 190, 191, 192, 194, 195, 197, 218, 1057]

rev_6_chg_cd = [71, 81, 91, 101, 101.001, 105, 109, 110.001, 129, 131, 132, 133, 141, 153, 155, 156, 157, 159, 182]

general_service = ['2', '5', '7', '9', '11', '26']
church = ['6']
residential = ['3', '4', '8', '33', '34']
general_service_pp = ['19']
residential_pp = ['13', '20']
lpd = ['21']
lpd_high_load = ['12']

df['new_rate'] = np.where(df['BI_RATE_SCHED'].isin(general_service), round((df['BI_USAGE']*.0725) + (df['BI_DMD_RDG']*12),2), \
                 np.where(df['BI_RATE_SCHED'].isin(church), round((df['BI_USAGE']*.091) + (df['BI_DMD_RDG']*4),2), \
                 np.where(df['BI_RATE_SCHED'].isin(residential), round((df['BI_USAGE']*.065) + (df['BI_DMD_RDG']*12),2), \
                 np.where(df['BI_RATE_SCHED'].isin(general_service_pp), round((df['BI_USAGE']*.0725) + (df['BI_DMD_RDG']*12),2), \
                 np.where(df['BI_RATE_SCHED'].isin(residential_pp), round((df['BI_USAGE']*.065) + (df['BI_DMD_RDG']*12),2), \
                 np.where(df['BI_RATE_SCHED'].isin(lpd_high_load), round((df['BI_USAGE']*.051) + (df['BI_DMD_RDG']*15.95),2), \
                 np.where(df['BI_RATE_SCHED'].isin(lpd), np.where(df['BI_USAGE']<=df['BI_DMD_RDG']*100, df['BI_USAGE']*.115, ((df['BI_USAGE']-100)*.105)+11.5) + (df['BI_DMD_RDG']*4.5), None)))))))

df['old_rate'] = np.where(df['BI_RATE_SCHED'].isin(general_service), round((df['BI_USAGE']*.128),2), \
                 np.where(df['BI_RATE_SCHED'].isin(church), round((df['BI_USAGE']*.128),2), \
                 np.where(df['BI_RATE_SCHED'].isin(residential), round((df['BI_USAGE']*.125),2), \
                 np.where(df['BI_RATE_SCHED'].isin(general_service_pp), round((df['BI_USAGE']*.128),2), \
                 np.where(df['BI_RATE_SCHED'].isin(residential_pp), round((df['BI_USAGE']*.125),2), \
                 np.where(df['BI_RATE_SCHED'].isin(lpd_high_load), round((df['BI_USAGE']*.051) + (df['BI_DMD_RDG']*15.95),2), \
                 np.where(df['BI_RATE_SCHED'].isin(lpd), np.where(df['BI_USAGE']<=df['BI_DMD_RDG']*100, df['BI_USAGE']*.115, ((df['BI_USAGE']-100)*.105)+11.5) + (df['BI_DMD_RDG']*4.5), None)))))))
   
df = df[~(df['BI_RATE_SCHED'] == '55')]

df_chg = pd.read_sql_query("SELECT BI_ACCT, BI_BILL_DT_TM, BI_CHG_CD, BI_CHG_AMT, BI_REV_YRMO FROM RPT43028.BI_HIST_CHG WHERE (BI_REV_YRMO >= 202111) AND (BI_REV_YRMO < 202311)", cnxncis)
df_chg['new_chg'] = df_chg['BI_CHG_AMT']
df_chg['old_chg'] = df_chg['BI_CHG_AMT']

df = df_chg.merge(df[['BI_ACCT', 'BI_BILL_DT_TM', 'BI_REV_CLASS_CD', 'BI_RATE_SCHED', 'new_rate', 'old_rate']], how='left', on=['BI_ACCT', 'BI_BILL_DT_TM'])
df['new_chg'] = np.where(df['BI_CHG_CD'] == 81, 0, df['new_chg'])
df['old_chg'] = np.where(df['BI_CHG_CD'] == 81, 0, df['old_chg'])
df['new_chg'] = np.where(df['BI_CHG_CD'] == 71, df['new_rate'], df['new_chg']).astype(float)
df['old_chg'] = np.where(df['BI_CHG_CD'] == 71, df['old_rate'], df['old_chg']).astype(float)

def plot_pred_rev(df, rev):
    df['BI_REV_YRMO'] = pd.to_datetime(df['BI_REV_YRMO'], format='%Y%m', errors='coerce').dropna()
    df = df.set_index('BI_REV_YRMO')
   
    model = ExponentialSmoothing(df['new_chg'], trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    pred = fit.forecast(15)
   
    model_old = ExponentialSmoothing(df['old_chg'], trend="add", seasonal="add", seasonal_periods=12)
    fit_old = model_old.fit()
    pred_old = fit_old.forecast(15)

   
    fig, ax = plt.subplots(figsize=(12, 6))
    #ax.plot(df.index, df['BI_CHG_AMT'], color='#4295FF', label='BI_CHG_AMT')
    ax.plot(df.index, df['new_chg'], color='#42F8FF', label='New Chg Amt')
    ax.plot(df.index, df['old_chg'], color='#ff0000', label='Old Chg Amt')
    ax.plot(pred.index, pred, linestyle='--', color='#42F8FF')
    ax.plot(pred_old.index, pred_old, linestyle='--', color='#ff0000')
    ax.set_title(f"Holt-Winter's Revenue Projection for Rev Class {rev}")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend(loc="upper left")
    plt.savefig(rf"XXXXX\Holt-Winter's Revenue Class {rev} Projection.svg")
   
    plt.plot(df.index, df['new_chg'])
    plt.show()
    plt.clf()
   
    return pred, pred_old


df_rev_1 = df[(df['BI_REV_CLASS_CD'] == 1) & (df['BI_CHG_CD'].isin(rev_1_chg_cd))].groupby(['BI_REV_YRMO']).agg({'BI_CHG_AMT':'sum', 'new_chg':'sum', 'old_chg':'sum'}).reset_index()
df_rev_2 = df[(df['BI_REV_CLASS_CD'] == 2) & (df['BI_CHG_CD'].isin(rev_2_chg_cd))].groupby(['BI_REV_YRMO']).agg({'BI_CHG_AMT':'sum', 'new_chg':'sum', 'old_chg':'sum'}).reset_index()
df_rev_4 = df[(df['BI_REV_CLASS_CD'] == 4) & (df['BI_CHG_CD'].isin(rev_4_chg_cd))].groupby(['BI_REV_YRMO']).agg({'BI_CHG_AMT':'sum', 'new_chg':'sum', 'old_chg':'sum'}).reset_index()
df_rev_6 = df[(df['BI_REV_CLASS_CD'] == 6) & (df['BI_CHG_CD'].isin(rev_6_chg_cd))].groupby(['BI_REV_YRMO']).agg({'BI_CHG_AMT':'sum', 'new_chg':'sum', 'old_chg':'sum'}).reset_index()

df_pred_1 = plot_pred_rev(df_rev_1, 1)
df_pred_2 = plot_pred_rev(df_rev_2, 2)
df_pred_4 = plot_pred_rev(df_rev_4, 4)
df_pred_6 = plot_pred_rev(df_rev_6, 6)
