import pandas as pd
import numpy as np

# Reload data after environment reset
prices = pd.read_parquet("data/US/eodhd_sp500_eod_20241016_20251016.parquet")
factors = pd.read_parquet("data/US/neutralized_eodhd_sp500_eod_all_factors_20241016_20251016.parquet")

# Merge factor and price data
data = pd.merge(factors, prices[['date', 'symbol', 'open', 'close']], on=['date', 'symbol'], how='inner')

# Identify alpha factor columns
alpha_cols = [col for col in factors.columns if 'alpha' in col.lower()]

# Strategy parameters
k = 50  # number of long positions
m = 50  # number of short positions
total_capital = 1_000_000  # total capital
cost_rate = 0.001  # 0.1% per trade side

# Container for all trades
trades = []

# Group by date and process each day
for date, df_day in data.groupby('date'):
    df_day = df_day.copy()
    df_day['alpha_score'] = df_day[alpha_cols].sum(axis=1)
    df_day.sort_values('alpha_score', ascending=False, inplace=True)

    longs = df_day.head(k)
    shorts = df_day.tail(m)
    num_positions = len(longs) + len(shorts)
    if num_positions == 0:
        continue

    capital_per_position = total_capital / num_positions

    # Process longs
    for _, row in longs.iterrows():
        open_price, close_price = row['open'], row['close']
        volume = np.floor(capital_per_position / open_price)
        if volume <= 0:
            continue
        pnl = (close_price - open_price) * volume
        cost = (open_price + close_price) * volume * cost_rate
        pnl_cost = pnl - cost
        trades.append({
            'symbol': row['symbol'],
            'entry_ts': pd.Timestamp(date).replace(hour=9, minute=30),
            'exit_ts': pd.Timestamp(date).replace(hour=16, minute=0),
            'entry_price': open_price,
            'exit_price': close_price,
            'volume': int(volume),
            'type': 'long',
            'exit_type': 'sell',
            'pnl': pnl,
            'pnl_cost': pnl_cost
        })

    # Process shorts
    for _, row in shorts.iterrows():
        open_price, close_price = row['open'], row['close']
        volume = np.floor(capital_per_position / open_price)
        if volume <= 0:
            continue
        pnl = (open_price - close_price) * volume
        cost = (open_price + close_price) * volume * cost_rate
        pnl_cost = pnl - cost
        trades.append({
            'symbol': row['symbol'],
            'entry_ts': pd.Timestamp(date).replace(hour=9, minute=30),
            'exit_ts': pd.Timestamp(date).replace(hour=16, minute=0),
            'entry_price': open_price,
            'exit_price': close_price,
            'volume': int(volume),
            'type': 'short',
            'exit_type': 'cover',
            'pnl': pnl,
            'pnl_cost': pnl_cost
        })

# Create DataFrame and save
trades_df = pd.DataFrame(trades)
output_path = "data/US/long_short_trade_log.parquet"
trades_df.to_parquet(output_path, index=False)
print('trade saved to ', output_path)
trades_df.head()
