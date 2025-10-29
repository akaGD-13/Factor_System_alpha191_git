import pandas as pd


def data_preprocessing(data):

    df = data.copy()
    df['adjust_factor'] = df['adjusted_close'] / df['close']
    df['open'] = df['open'] * df['adjust_factor']
    df['high'] = df['high'] * df['adjust_factor']
    df['low'] = df['low'] * df['adjust_factor']
    df['volume'] = df['volume'] / df['adjust_factor']
    df = df.rename(columns={
        'close': 'unadjusted_close', 
        'adjusted_close': 'close', 
        'symbol': 'ts_code',
        'date': 'trade_date'})
    df['vwap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['pct_chg'] = df.groupby('ts_code')['close'].pct_change()

    return df

if __name__ =='__main__':
    df = pd.read_parquet('data/US/eodhd_sp500_eod_20241016_20251016.parquet')
    print(data_preprocessing(df).head())
