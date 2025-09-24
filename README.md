# Alpha191 Project

Author: Trove Quant 
Date: Sep 2025

---

<details>
<summary><h2>data</h2></summary>

### data/sliced/ (preprocessed datasets)
- **all_stocls_daily_factors_300_....parquet**  
  Daily factor panel for ~300 stocks.  
  <br>Columns: ts_code, date, factors
- **all_stocls_daily_price_vol_....parquet**  
  Daily price-volume data panel for ~300 stocks.  
  <br>Columns: ts_code, date, price-volumes
- **neutralize_data_sliced_300_....parquet**  
  industry, size for factor neutralization.

### data/benchmark.csv
Benchmark index/returns for evaluation (HS300)

### data/industry_data.csv
Industry classification mapping (ticker → industry/code).

</details>

<details>
<summary><h2>Core Scripts</h2></summary>

### factor_calculation_main.py
End-to-end runner for the factor pipeline.  
Inputs: files under `data/sliced/`  
Outputs: factor panels

### factor_calculators.py
Library of factor functions (e.g., momentum, quality, value, volatility).  
Reusable APIs called by `factor_calculation_main.py`.

### factor_calculation_example.py
Minimal example showing how to load data, compute one/two factors, and save outputs.

### one_factor_ic.py
Computes and graph Information Coefficient for a single factor

### data_slicing.py
Transforms raw market data → tidy panels under `data/sliced/` (filtering, aligning calendars, handling missing values).

### 策略_20250730.py
Example strategy/backtest using selected factors to form portfolios.
<br> outputs: statistics and net values graph

</details>

<details>
<summary><h2>Getting Started</h2></summary>

### 1) Environment (unfinished)
- Python ≥ 3.10  
- Suggested packages: `pandas`, `numpy`, `scipy`, `pyarrow` or `fastparquet`, `matplotlib`, `seaborn` (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
