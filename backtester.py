import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

settings = dict(
    trades_path='data/US/long_short_trade_log.parquet',
    graph_option=True,
    initial_equity=1.0,  # Net value base
    risk_free_rate=0.0,
    trading_days_per_year=365,
)

def _to_utc_day(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.floor("D")

def _max_drawdown(series: pd.Series):
    series = series.copy()
    running_max = series.cummax()
    dd = series / running_max - 1
    max_dd = dd.min()
    return dd, max_dd

def compute_daily_returns(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df = df[df["entry_price"] > 1e-3].copy()
    df["ret"] = df["pnl"] / df["entry_price"] / df['volume']
    df["exit_day"] = _to_utc_day(df["exit_ts"])

    # Create daily returns table
    ret_all = df.groupby("exit_day")["ret"].mean().rename("ret_all")
    ret_long = df[df["type"].str.lower() == "long"].groupby("exit_day")["ret"].mean().rename("ret_long")
    ret_short = df[df["type"].str.lower() == "short"].groupby("exit_day")["ret"].mean().rename("ret_short")

    daily_returns_df = pd.concat([ret_all, ret_long, ret_short], axis=1).fillna(0)
    print(daily_returns_df)
    return daily_returns_df

def plot_curves(daily_returns_df: pd.DataFrame):
    # Cumulative return
    cum_all = (1 + daily_returns_df["ret_all"]).cumprod()
    cum_long = (1 + daily_returns_df["ret_long"]).cumprod()
    cum_short = (1 + daily_returns_df["ret_short"]).cumprod()

    # Drawdown
    dd_all, _ = _max_drawdown(cum_all)
    dd_long, _ = _max_drawdown(cum_long)
    dd_short, _ = _max_drawdown(cum_short)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot cumulative return
    axes[0].plot(cum_all, label="All Trades", color="black")
    axes[0].plot(cum_long, label="Long Trades", color="tab:blue")
    axes[0].plot(cum_short, label="Short Trades", color="tab:orange")
    axes[0].set_title("Cumulative Return")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    # Plot drawdowns
    axes[1].plot(dd_all * 100, label="All", color="black")
    axes[1].plot(dd_long * 100, label="Long", color="tab:blue")
    axes[1].plot(dd_short * 100, label="Short", color="tab:orange")
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("return_backtest_report.png", dpi=120)
    plt.show()

def main():
    path = Path(settings["trades_path"])
    trades = pd.read_parquet(path)

    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], errors="coerce", utc=True)

    # Compute daily average returns
    daily_returns_df = compute_daily_returns(trades)

    # Plot
    if settings["graph_option"]:
        plot_curves(daily_returns_df)

if __name__ == "__main__":
    main()
