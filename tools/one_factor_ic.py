import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os
np.seterr(divide='ignore', invalid='ignore')

def one_factor_ic(prices_data: pd.DataFrame, factor_data: pd.DataFrame, factor_name: str, graph=True) -> float:
    # ensure sorted (your original line didn't assign)

    prices = prices_data.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
    factor = factor_data.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
    prices = prices.sort_values(['symbol', 'date']).copy()

    # next-day return proxy prepared from prices
    prices['pct_chg'] = (
        prices.groupby('symbol')['close']
              .pct_change()
              .shift(-1)  # align next day's return with today's row
    )

    alpha_data = factor[['date', 'symbol', factor_name]].dropna()
    calculate_df = pd.merge(alpha_data,
                            prices[['date', 'symbol', 'pct_chg']],
                            on=['date', 'symbol'],
                            how='left').sort_values('date')

    date_list = calculate_df['date'].unique()
    ic = pd.DataFrame(index=date_list, columns=[factor_name], dtype=float)

    # cross-sectional IC each day vs. next day's returns
    for i, date in enumerate(date_list[:-1]):
        daily = calculate_df[calculate_df['date'] == date].drop(columns='pct_chg')
        nxt   = calculate_df[calculate_df['date'] == date_list[i+1]][['symbol', 'pct_chg']]
        merged = daily.merge(nxt, on='symbol', how='left').dropna()
        ic.loc[date, factor_name] = merged[factor_name].corr(merged['pct_chg'], method='pearson')

    # paths
    output_path = 'back_test/ic_backtest_results'
    os.makedirs(output_path, exist_ok=True)
    ic_csv_path = os.path.join(output_path, 'factor_mean_ic.csv')

    # plot (optional)
    if graph:
        # try to convert to datetime for nicer x-axis; fall back if already datetime
        try:
            x = pd.to_datetime(ic.index, format='%Y%m%d')
        except Exception:
            x = pd.to_datetime(ic.index)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, ic[factor_name])
        ax.axhline(0, lw=1, color='k')
        ax.set_xlabel('Date'); ax.set_ylabel('IC'); ax.set_title(f'{factor_name} daily IC')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'ic_graph_{factor_name}.png'))
        plt.close(fig)

    mean_ic = float(ic[factor_name].mean())

    # ---- update/create the CSV of mean ICs ----
    new_row = pd.DataFrame([{'factor': factor_name, 'mean_ic': mean_ic}])

    if os.path.exists(ic_csv_path):
        try:
            agg = pd.read_csv(ic_csv_path)
        except Exception:
            # if the existing file is corrupted/unreadable, start fresh
            agg = pd.DataFrame(columns=['factor', 'mean_ic'])
    else:
        agg = pd.DataFrame(columns=['factor', 'mean_ic'])

    if 'factor' not in agg.columns or 'mean_ic' not in agg.columns:
        agg = pd.DataFrame(columns=['factor', 'mean_ic'])

    mask = agg['factor'] == factor_name
    if mask.any():
        agg.loc[mask, 'mean_ic'] = mean_ic
    else:
        agg = pd.concat([agg, new_row], ignore_index=True)

    agg.to_csv(ic_csv_path, index=False)

    return mean_ic

if __name__ == '__main__':
    price_vol_data_file_path = 'data/US/eodhd_sp500_eod_20241016_20251016.parquet'
    factor_output_path = 'data/US/neutralized_eodhd_sp500_eod_all_factors_20241016_20251016.parquet'
    ic_output_path = 'back_test/ic_backtest_results'

    data = pd.read_parquet(price_vol_data_file_path)
    factors = pd.read_parquet(factor_output_path)  # make sure that first two columns is trade_date and ts_code
    alpha_list = factors.columns[2:]
    ic_form = pd.DataFrame(index=alpha_list.values)
    ic_form['ic'] = 0
    for factor_name in tqdm.tqdm(alpha_list):
        single_ic = one_factor_ic(data, factors, factor_name)
        ic_form.loc[factor_name,'ic'] = single_ic
    print(ic_form)


