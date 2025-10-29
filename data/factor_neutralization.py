import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# ---------- Paths ----------
factor_data_PATH  = 'data/US/eodhd_sp500_eod_all_factors_20241016_20251016.parquet'
shares_data_PATH  = 'data/US/sp500_eps_panel.parquet'
industry_PATH     = 'data/US/sp500_industry_20251016.csv'
market_data_PATH  = 'data/US/eodhd_sp500_eod_20241016_20251016.parquet'
OUTPUT_PATH       = 'data/US/neutralized_eodhd_sp500_eod_all_factors_20241016_20251016.parquet'

# ---------- Read data ----------
factor      = pd.read_parquet(factor_data_PATH)
shares_raw  = pd.read_parquet(shares_data_PATH)
industry    = pd.read_csv(industry_PATH)
market_data = pd.read_parquet(market_data_PATH)

# If the shares file still has original names:
factor = factor.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
shares = shares_raw.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
industry = industry.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
market_data = market_data.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})

# ---------- Clean symbol naming ----------
for df in (factor, shares, industry, market_data):
    df['symbol'] = df['symbol'].str.replace('-', '.', regex=False)

# ---------- Ensure date dtype ----------
for df in (factor, shares, market_data):
    df['date'] = pd.to_datetime(df['date'])

# ---------- Compute log market cap ----------
# Align shares with closes by date & symbol, then ffill per symbol
mktcap_base = (market_data[['symbol', 'date', 'close']]
               .merge(shares[['symbol', 'date', 'shares']], on=['symbol', 'date'], how='left')
               .sort_values(['symbol', 'date']))
mktcap_base['shares'] = mktcap_base.groupby('symbol')['shares'].ffill()  # ffill within symbol
mktcap_base['market_cap'] = mktcap_base['shares'] * mktcap_base['close']
mktcap_base['log_mktcap'] = np.log(mktcap_base['market_cap'].replace(0, np.nan))

# ---------- Join market cap & sector to factor panel ----------
factor = (factor.merge(mktcap_base[['symbol', 'date', 'log_mktcap', 'shares']], on=['symbol', 'date'], how='left')
                .merge(industry[['symbol', 'sector']], on='symbol', how='left')
                .sort_values(['symbol', 'date']))

# In case some dates precede the first shares point for a symbol:
factor['log_mktcap'] = factor.groupby('symbol')['log_mktcap'].ffill()

# ---------- Identify factor columns ----------
exclude_cols = {'date', 'symbol', 'log_mktcap', 'sector', 'shares'}
factor_cols = [c for c in factor.columns if c not in exclude_cols]

# ---------- Cross-sectional neutralization (per date) ----------
def neutralize_by_date(df_date):
    res = df_date.copy()
    X = df_date[['log_mktcap']].replace([np.inf, -np.inf], np.nan)
    # If no valid mktcap, bail
    if X['log_mktcap'].notna().sum() < 5:
        res[factor_cols] = np.nan
        return res
    for fac in factor_cols:
        y = df_date[fac].replace([np.inf, -np.inf], np.nan)
        mask = y.notna() & X['log_mktcap'].notna()
        if mask.sum() < 5:
            res[fac] = np.nan
            continue
        model = LinearRegression().fit(X.loc[mask], y.loc[mask])
        res.loc[mask, fac] = y.loc[mask] - model.predict(X.loc[mask])
        res.loc[~mask, fac] = np.nan
    return res

# neutralized = factor.groupby('date', group_keys=False).apply(neutralize_by_date)
neutralized = factor

# ---------- Winsorize residuals (per date) ----------
def winsorize_series(s, p=0.01):
    if s.notna().sum() < 10:  # guard small cross-sections
        return s
    lo, hi = s.quantile([p, 1-p])
    return s.clip(lo, hi)

for fac in factor_cols:
    neutralized[fac] = neutralized.groupby('date')[fac].transform(winsorize_series)

# ---------- Standardize within sector (per date) ----------
for fac in factor_cols:
    # zscore returns array; ensure we keep NaNs for small groups
    neutralized[fac] = neutralized.groupby(['date', 'sector'])[fac] \
                                  .transform(lambda x: zscore(x, nan_policy='omit'))

# ---------- Done ----------
neutralized = neutralized.drop(columns=['log_mktcap', 'shares', 'sector'])
print(neutralized.head())
neutralized.to_parquet(OUTPUT_PATH)
print('Output saved in', OUTPUT_PATH)
