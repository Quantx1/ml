# NSE tier seed files

One symbol per line. `#` is a comment. Symbols are NSE tickers **without** the `.NS` suffix (yfinance suffix is added at fetch time).

Files:
- `nifty50.txt`   — Nifty 50
- `nifty100.txt`  — Nifty 100 (Nifty 50 + Nifty Next 50)
- `nifty250.txt`  — Nifty 100 + Nifty Midcap 150
- `nifty500.txt`  — Nifty 500
- `nse_all.txt`   — every liquid NSE equity we rank (superset)

Refresh quarterly from NSE's published index constituents CSVs:
- https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv
- https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv
- https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv
- https://nsearchives.nseindia.com/content/indices/ind_nifty_total_marketlist.csv
