from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

def adf_test(series):
    r=adfuller(series.dropna(),autolag='AIC')
    return {'test_statistic':r[0],'p_value':r[1],'used_lags':r[2],'n_observations':r[3],'critical_values':r[4]}
def save_eda(df,target,out):
    out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,6)); df[target].plot(ax=ax); ax.set(title='Hourly appliance energy use',xlabel='Date',ylabel='Appliance energy use'); fig.tight_layout(); fig.savefig(out/'01_hourly_series.png',dpi=300); plt.close(fig)
    for name,s,title,xlabel in [('02_hour_of_day',df[target].groupby(df.index.hour).mean(),'Average use by hour of day','Hour'),('03_day_of_week',df[target].groupby(df.index.dayofweek).mean(),'Average use by day of week','Day of week')]:
        fig,ax=plt.subplots(figsize=(10,5)); s.plot(ax=ax,marker='o'); ax.set(title=title,xlabel=xlabel,ylabel='Mean appliance energy use'); fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); plot_acf(df[target].dropna(),lags=168,ax=ax); ax.set_title('ACF of hourly appliance use'); fig.tight_layout(); fig.savefig(out/'04_acf.png',dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); plot_pacf(df[target].dropna(),lags=72,ax=ax,method='ywm'); ax.set_title('PACF of hourly appliance use'); fig.tight_layout(); fig.savefig(out/'05_pacf.png',dpi=300); plt.close(fig)
    diff=df[target].diff().dropna(); fig,ax=plt.subplots(figsize=(12,5)); plot_acf(diff,lags=168,ax=ax); ax.set_title('ACF after first differencing'); fig.tight_layout(); fig.savefig(out/'06_acf_difference.png',dpi=300); plt.close(fig); return diff
