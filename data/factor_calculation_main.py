import pandas as pd
import factor_calculators
import inspect
import data_preprocessing as d_p


# price_vol_data_file_path = 'data/sliced/all_stocls_daily_price_vol_sliced_300_20180101_20201231'
# factor_output_path = 'data/sliced/all_stocls_daily_factors_300_20180101_20201231'

price_vol_data_file_path = 'data/US/eodhd_sp500_eod_20241016_20251016.parquet'
factor_output_path = 'data/US/eodhd_sp500_eod_all_factors_20241016_20251016.parquet'

data = pd.read_parquet(price_vol_data_file_path)
data['symbol'] = data['symbol'].str.replace('-', '.', regex=False)
data = d_p.data_preprocessing(data)

# print(data.columns) TESTING Individual Facotrs ==============================
alpha = factor_calculators.alpha25(data)
print(alpha)
exit(0)

all_factor_data = data[['trade_date', 'ts_code']]

factor_funcs = [
        (name, fn) 
        for name, fn in inspect.getmembers(factor_calculators, inspect.isfunction)
        if name.startswith("alpha")
    ]

for name, fn in factor_funcs:
    setting = {
        
        'factor_name': name
    }

    factor_data = fn(data.copy())
    # wide_df: index=trade_date, columns=ts_code, values=alpha21
    # long_df = (
    #     factor_data
    #     .stack()                # Series with MultiIndex (trade_date, ts_code)
    #     .rename(setting['factor_name'])      # name the series
    #     .reset_index()          # turn index levels into columns
    # )

    # # rename columns if needed
    # long_df.columns = ['trade_date','ts_code',setting['factor_name']]

    all_factor_data = pd.merge(all_factor_data, factor_data, on=['trade_date', 'ts_code'], how='left')


all_factor_data.to_parquet(factor_output_path)
print('factors data saved in', factor_output_path + ', ' + str(len(factor_funcs))+' factors in total.')