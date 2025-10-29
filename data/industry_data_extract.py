import pandas as pd
import yfinance as yf
import time

EXCEL_PATH = "data/US/sp500_weights_20251016.xlsx"
OUTPUT_PATH = "data/US/sp500_industry_20251016.csv"

def to_yahoo_symbol(s: str) -> str:
    """Convert common non-Yahoo forms to Yahoo’s convention."""
    # Class shares: BRK.B -> BRK-B, BF.B -> BF-B, etc.
    return s.replace(".", "-").strip().upper()

def from_yahoo_symbol(s: str) -> str:
    """Back to your original dot-notation if you prefer."""
    return s.replace("-", ".")

def fetch_sector_industry(y_symbol: str) -> tuple[str | None, str | None]:
    """
    Return (sector, industry) from yfinance for a Yahoo-form symbol.
    Uses .get_info(); handle throttling and missing fields gracefully.
    """
    try:
        tk = yf.Ticker(y_symbol)
        info = tk.get_info()  # dict; may be slow/throttled
        sector = info.get("sector")
        industry = info.get("industry")
        # Some tickers expose 'longBusinessSummary' but not industry/sector -> return None
        return sector, industry
    except Exception:
        return None, None

def build_symbol_industry_table(excel_path: str) -> pd.DataFrame:
    # 1) read your Excel, normalize column names and symbols
    w = pd.read_excel(excel_path, index_col=0)
    w.columns = [c.strip().lower() for c in w.columns]
    if "symbol" not in w.columns:
        raise ValueError("Excel must contain a 'Symbol' column.")
    syms_raw = w["symbol"].astype(str).str.strip()

    # 2) map to Yahoo symbols
    syms_yahoo = syms_raw.map(to_yahoo_symbol)

    # 3) de-duplicate to keep requests light
    unique_yahoo = pd.unique(syms_yahoo)

    # 4) fetch sector/industry with polite pacing to avoid throttling
    rows = []
    for i, ys in enumerate(unique_yahoo, 1):
        print(str(i), str(ys))
        sector, industry = fetch_sector_industry(ys)
        rows.append({"yahoo_symbol": ys, "sector": sector, "industry": industry})
        # light sleep to be nice to Yahoo; tune as needed
        time.sleep(0.15)

    meta = pd.DataFrame(rows)

    # 5) join back to your original list, restore your original symbol string
    out = (
        pd.DataFrame({"symbol": syms_raw, "yahoo_symbol": syms_yahoo})
        .merge(meta, on="yahoo_symbol", how="left")
        .drop(columns=["yahoo_symbol"])
    )

    # Show a quick preview
    print(out.head(10))

    return out

if __name__ == "__main__":
    df_symbol_industry = build_symbol_industry_table(EXCEL_PATH)
    df_symbol_industry.to_csv(OUTPUT_PATH)
