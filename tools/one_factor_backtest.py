import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def data_read_merge(prices, factor_data, setting):
    prices['date'] = pd.to_datetime(prices['date'])
    factor_data['date'] = pd.to_datetime(factor_data['date'])

    # filter by dates
    prices = prices[(prices['date'] >= pd.to_datetime(setting['start_date'])) & (prices['date'] <= pd.to_datetime(setting['end_date']))]
    factor_data = factor_data[(factor_data['date'] >= pd.to_datetime(setting['start_date'])) & (factor_data['date'] <= pd.to_datetime(setting['end_date']))]
    # merge
    merged_df = pd.merge(prices[['date', 'symbol', 'open', 'close']], factor_data[['date', 'symbol', setting['factor_name']]], on=['date', 'symbol'], how='left')

    return merged_df

def factor_standardization(df: pd.DataFrame, setting) -> pd.DataFrame:
    """
    Standardize factor cross-sectionally by date:
    - compute z-score
    - assign layer by quantile (1...layers)
    """
    fname = setting['factor_name']
    # drop rows with missing factor
    df = df.dropna(subset=[fname])
    # compute z-score per date
    df['s_' + fname] = df.groupby('date')[fname].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    # assign layers 1...layers
    df['layer'] = df.groupby('date')['s_' + fname].transform(
        lambda x: pd.qcut(x, setting['layers'], labels=False, duplicates='drop') + 1
    )
    return df

def generate_net_values(df: pd.DataFrame, setting) -> pd.DataFrame:
    """
    For each date & layer, compute equal-weight daily return and net value.
    Also compute benchmark and valid count.
    """
    # compute daily return per stock: (close - open)/open
    df['ret'] = (df['close'] - df['open']) / df['open']
    # group returns by date & layer
    grp = df.groupby(['date','layer'])['ret'].mean().unstack('layer')
    # benchmark: equal-weight across all symbols
    bench = df.groupby('date')['ret'].mean()
    # count valid stocks per date
    count = df.groupby('date')['symbol'].nunique()
    # compute net values
    nav = (1 + grp).cumprod()
    nav.columns = [f'layer_{int(c)}' for c in nav.columns]
    nav['benchmark'] = (1 + bench).cumprod()
    # add count
    nav['count'] = count
    return nav.reset_index()


def graph_drawing(nav_df: pd.DataFrame, setting):
    """
    Plot three panels:
      1) Net‐value curves for each layer + benchmark
      2) Count of valid stocks over time
      3) Bar chart of final net value by layer

    nav_df must have columns: ['date','layer_1',…,'layer_N','benchmark','count']
    """
    # prepare
    dates  = pd.to_datetime(nav_df['date'])
    layers = [c for c in nav_df.columns if c.startswith('layer_')]

    # make a 3‐row figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,10))

    # ─── 1) Top: net‐value curves ───
    for col in layers:
        ax1.plot(dates, nav_df[col], label=col)
    # benchmark in black
    ax1.plot(dates, nav_df['benchmark'],
             color='black', linewidth=2, label='benchmark')
    ax1.set_ylabel('Net Value')
    ax1.legend(ncol=3, fontsize='small')

    # ─── 2) Middle: valid count ───
    ax2.plot(dates, nav_df['count'])
    ax2.set_ylabel('Valid Count')

    # ─── 3) Bottom: final net‐value by layer ───
    # extract final date’s values
    final_nav = nav_df[layers].iloc[-1]
    # convert 'layer_1' → 1, etc.
    layer_nums = [int(c.split('_')[1]) for c in layers]
    ax3.plot(layer_nums, final_nav)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Final Net Value')
    ax3.set_xticks(layer_nums)

    # layout & save
    plt.tight_layout()
    os.makedirs(setting['output_path'], exist_ok=True)
    plt.savefig(os.path.join(setting['output_path'], 'layer_backtest.png'))
    plt.show()


def layer_backtest(prices, factor_data, setting) -> pd.DataFrame:
    df = data_read_merge(prices, factor_data, setting)
    df = factor_standardization(df, setting)
    nav_df = generate_net_values(df, setting)
    graph_drawing(nav_df, setting)

    return nav_df


if __name__ == '__main__':
    setting = {
        'factor_name' : 'volume',
        'factor_data_file_path' : 'stock_data/weekly/sp500_price_vol_adjusted',
        'prices_data_file_path' : 'stock_data/weekly/sp500_price_vol_adjusted',
        'output_path' : 'back_test/layer_backtest_results/',
        'layers' : 10,  # number of layers
        'period': 1,  # holding period
        'min_number': 10, # number of minimum valid stocks (that has not-null values)
        'start_date' : '2018-01-01', 
        'end_date' : '2019-12-31'
    }
    json_path = 'json/one_factor_backtest.json'

    if len(sys.argv) >= 2 and sys.argv[1]:
        json_path = sys.argv[1]
    else:
        print('One Factor Backtest use default json:', json_path)

    # read the json file
    setting = json.load(open(json_path, 'r'))
    # read data
    prices = pd.read_parquet(setting['prices_data_file_path'])
    factor_data = pd.read_parquet(setting['factor_data_file_path'])
    result = layer_backtest(prices, factor_data, setting)
    # save result
    out_file = os.path.join(setting['output_path'], 'net_values.parquet')
    result.to_parquet(out_file, index=False)
    print(f"Layer backtest result saved to {out_file}")
