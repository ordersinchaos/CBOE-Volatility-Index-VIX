import datetime as dt
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
np.set_printoptions(threshold=10000)

# Calculate dK
def f(group):
    new = group.copy()
    new.iloc[1:-1] = np.array((group.iloc[2:] - group.iloc[:-2]) / 2)
    new.iloc[0] = group.iloc[1] - group.iloc[0]
    new.iloc[-1] = group.iloc[-1] - group.iloc[-2]
    return new

# Calculate volatility for both near-term and next-term options
def g(group):
    days = np.array(group['Days'])
    sigma2 = np.array(group['sigma2'])
    if len(days)==1:
        T1 = 0
        T2 = days[0]
        sigma_T1 = 0
        sigma_T2 = sigma2[0]
    else:
        if days.min() <= 27:
            T1 = days[days <= 27].max()
        else:
            T1 = days.min()

        T2 = days[days > T1].min()

        sigma_T1 = sigma2[days == T1][0]
        sigma_T2 = sigma2[days == T2][0]

    
    return pd.DataFrame([{'T1' : T1, 'T2' : T2, 'sigma2_T1' : sigma_T1, 'sigma2_T2' : sigma_T2}])
    
def VIX_with_Divdent(key):
    # obtain one tading date
    df = pd.read_csv("C:/Users/Cheetah/Desktop/5202/assignment/Assignment1/data.csv")
    df['date'] = pd.to_datetime(df['date'])
    v1 = df.date.unique()
    df1 = df[df['date'] == v1[key]]
    
    #re-order
    df1.index = range(0, len(df1)) 
    c=[]
    d=[]
    for i in range(len(df1)):
        a=df1.name[i]
        c.append(a.split()[2])
        d.append(c[i][1:])
    df1['Strike']=d   
    df1['Strike']=pd.to_numeric(df1['Strike']) 
    df1['maturity'] = pd.to_datetime(df1['maturity'])
    df1['date'] = pd.to_datetime(df1['date'])
    df1['DAYS']=df1['maturity']-df1['date']
    # Function to convert days to internal timedelta format
    df1['DAYS']=(df1['DAYS']/ np.timedelta64(1, 'D')).astype(int)
    df1['Premium'] = (df1['bid'] + df1['ask']) / 2
    
    raw_options=pd.DataFrame()
    df4=df1[df1['type']=='Call'] #extract all call
    df4.index=range(0,len(df4))
    df2=df1[df1['type']=='Put'] #exctract all put
    df2.index=range(0,len(df2))
    
    #rename
    raw_options['Expiration']=df4["maturity"]
    raw_options["Days"]=df4['DAYS']
    raw_options['Strike']=df4['Strike']
    raw_options['Call Bid']=df4['bid']
    raw_options['Call Ask']=df4['ask']
    raw_options['Put Bid']=df2['bid']
    raw_options['Put Ask']=df2['ask']
    raw_options["Date"]=df4["date"]
    raw_options["Name"]=df4["name"]

    
    # Since VIX is computed for the date of option quotations, we do not really need Expiration
    raw_options = raw_options.set_index(['Date','Days','Strike']).drop('Expiration', axis = 1)
    raw_options = raw_options.reset_index().set_index(['Date','Days','Strike']).sort_index()
 
    
    raw_options["C"]=(raw_options["Call Bid"]+raw_options["Call Ask"])/2
    raw_options["P"]=(raw_options["Put Bid"]+raw_options["Put Ask"])/2
    raw_options["C-P"]=(raw_options["C"]-raw_options["P"]).abs()

    raw_options['min'] = raw_options['C-P'].groupby(level = ['Days']).transform(lambda x: x == x.min())
    #delete dulipicate min
    raw_options.drop_duplicates(['C-P','min'],'first',inplace=True)
    #delete useless columes
    raw_options = raw_options.drop('Call Bid', axis = 1)
    raw_options = raw_options.drop('Call Ask', axis = 1)
    raw_options = raw_options.drop('Put Bid', axis = 1)
    raw_options = raw_options.drop('Put Ask', axis = 1)
    
    # Leave only at-the-money optons
    forward = raw_options[raw_options['min'] == 1].reset_index()

    # Compute the implied forward
    forward['Forward'] = forward['C-P'] * np.exp(3 * forward['Days'] / 36500)

    forward['Forward'] += forward['Strike']
    forward['Forward']=pd.to_numeric(forward['Forward'])
    forward = forward.set_index(['Date','Days'])[['Forward']]
    # Merge options with implied forward price
    left = raw_options.reset_index().set_index(['Date','Days'])
    raw_options = pd.merge(left, forward, left_index = True, right_index = True)#加入Forward列

    # Compute at-the-money strike
    mid_strike = raw_options[raw_options['Strike'] < raw_options['Forward']]['Strike'].groupby(level = ['Date','Days']).max()
    mid_strike = pd.DataFrame({'Mid Strike' : mid_strike})
    raw_options = pd.merge(raw_options, mid_strike, left_index = True, right_index = True)#加入mid_strike列
    raw_options['Premium']=raw_options['P']
    for i in range(len(raw_options)):
        if raw_options['Strike'][i] <=raw_options['Mid Strike'][i]:
            raw_options['Premium'][i]=raw_options['P'][i]
        elif raw_options['Strike'][i] ==raw_options['Mid Strike'][i]:
            raw_options['Premium'][i]=(raw_options['P'][i]+raw_options['C'][i])/2
        else:
            raw_options['Premium'][i]=raw_options['C'][i]
    raw_options['dK'] =raw_options.groupby(level = ['Date','Days'])['Strike'].apply(f)   #这里f

    raw_options['Rate']=0.03
    contrib = raw_options.reset_index().set_index(['Date'])

    contrib['sigma2'] = contrib['dK'] / contrib['Strike'] ** 2
    contrib['sigma2'] *= contrib['Premium'] * np.exp(contrib['Rate'] * contrib['Days'] / 36500)
    
    # Sum up contributions from all strikes
    sigma2 = contrib.groupby(['Date', 'Days'])[['sigma2']].sum() * 2

    # Merge at-the-money strike and implied forward
    sigma2['Mid Strike'] = mid_strike
    sigma2['Forward'] = forward

    # Compute variance for each term
    sigma2['sigma2'] -= (sigma2['Forward'] / sigma2['Mid Strike'] - 1) ** 2
    sigma2['sigma2'] /= sigma2.index.get_level_values(1).astype(float) / 365
    sigma2 = sigma2[['sigma2']]
    if sigma2.sigma2[0]==np.inf:
        sigma2.sigma2[0]=0


    two_sigmas = sigma2.reset_index().groupby('Date').apply(g).groupby(level='Date').first() #这里g
    df = two_sigmas.copy()

    for t in ['T1', 'T2']:
        # Convert to fraction of the year
        df['days_' + t] = df[t].astype(float) / 365
        # Convert to miutes
        df[t] = (df[t] - 1) * 1440. + 510 + 930

    df['sigma2_T1'] = df['sigma2_T1'] * df['days_T1'] * (df['T2'] - 30. * 1440.)
    df['sigma2_T2'] = df['sigma2_T2'] * df['days_T2'] * (30. * 1440. - df['T1'])
    df['VIX'] = ((df['sigma2_T1'] + df['sigma2_T2']) / (df['T2'] - df['T1']) * 365. / 30.) ** .5 * 100

    VIX = df[['VIX']]
    return VIX.VIX[0]

df = pd.read_csv("C:/Users/Cheetah/Desktop/5202/assignment/Assignment1/data.csv")
df['date'] = pd.to_datetime(df['date'])
v1 = df.date.unique()  #obtain unique trading dates 
y=[]
for i in range(len(v1)):
    y.append(VIX_with_Divdent(i))
Vix=pd.DataFrame()
x=v1[0:len(y)]
Vix["Date"]=x
Vix["VIX"]=y
plt.xlabel("Date",fontsize=13)
plt.ylabel("Vix",fontsize=13)
plt.title("ETF50 VIX")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  
plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=4))
plt.plot(x,y,label='calculate Vix')
plt.show()
