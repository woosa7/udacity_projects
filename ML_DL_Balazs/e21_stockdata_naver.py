import pandas as pd
import matplotlib.pyplot as plt

# KT
code = '030200'
url_tmpl = 'http://finance.naver.com/item/sise_day.nhn?code=%s&page=%d'
npages = 480

df_base = pd.DataFrame({
    'code':[code],
    'date':['2018-08-01'],
    'close':[0],
    'change':[0],
    'open':[0],
    'high':[0],
    'low':[0],
    'volume':[0]
}, columns=['date', 'close', 'change', 'open', 'high', 'low', 'volume'])
print(df_base)

for p in range(npages, 0, -1):
    url = url_tmpl % (code, p)
    dfs = pd.read_html(url)
    df_price = dfs[0]
    df_price = df_price.dropna()
    df_price.columns = ['date', 'close', 'change', 'open', 'high', 'low', 'volume']
    df_price = df_price[1:]
    df_price = df_price.replace('\.', '-', regex=True)
    # df_price['date'] = pd.to_datetime(df_price['date'])
    int_cols = ['close', 'change', 'open', 'high', 'low', 'volume']
    df_price[int_cols] = df_price[int_cols].astype('int', errors='ignore')
    df_price['code'] = code

    df_base = pd.concat([df_base, df_price])

    if p % 10 == 0:
        print('%d,' % p)


df = df_base.sort_values(by='date')
df = df[df.close != 0]
print('')
print(df)
print(df.shape)

df.to_csv("data/stock_price.csv", index=False)

vclose = df['close']
daily_returns = (vclose/vclose.shift(1)) - 1
daily_returns.hist(bins=100)
plt.show()
