'''
collect_stock_returns.py
========================
Collects PYPL historical stock prices via the official NASDAQ API and computes fractional returns.
Only fetches from the NASDAQ API if the latest timestamp in pypl_returns.csv is not today.
'''

import csv
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Base paths
CODE_DIR = Path(__file__).parent.resolve()
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"


def fetch_stock_prices_from_nasdaq_api(ticker: str, min_ts: datetime = None, max_ts: datetime = None) -> list:
    '''
    Fetches historical stock prices directly from NASDAQ official API (api.nasdaq.com).
    Returns list of tuples sorted chronologically: (timestamp_utc, close)
    '''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }

    from_date = (min_ts - timedelta(days=5)).strftime("%Y-%m-%d") if min_ts else "2016-08-12"
    to_date = max_ts.strftime("%Y-%m-%d") if max_ts else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    url = f"https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass=stocks&fromdate={from_date}&todate={to_date}&limit=9999"
    log.info(f"[{ticker}] Querying NASDAQ API: {url} ...")

    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            rows_data = data.get("data", {}).get("tradesTable", {}).get("rows", [])
            if rows_data:
                parsed_rows = []
                for item in rows_data:
                    d_str = item.get("date", "").strip()
                    c_str = item.get("close", "").replace("$", "").replace(",", "").strip()
                    if not d_str or not c_str or c_str == "N/A":
                        continue
                    try:
                        dt = datetime.strptime(d_str, "%m/%d/%Y").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue

                    if min_ts is not None and dt <= min_ts:
                        continue
                    if max_ts is not None and dt > max_ts:
                        continue

                    try:
                        close_val = float(c_str)
                    except ValueError:
                        continue

                    parsed_rows.append((dt, close_val))

                # Deduplicate and sort chronologically
                seen_dates = set()
                dedup_rows = []
                for dt, close_val in parsed_rows:
                    dt_key = dt.strftime("%Y-%m-%d")
                    if dt_key not in seen_dates:
                        seen_dates.add(dt_key)
                        dedup_rows.append((dt, close_val))

                dedup_rows.sort(key=lambda x: x[0])
                log.info(f"[{ticker}] Successfully fetched {len(dedup_rows)} records from NASDAQ API.")
                return dedup_rows
    except Exception as e:
        log.warning(f"[{ticker}] NASDAQ API query encountered an error: {e}")

    return []


def compute_stock_returns(rows: list, prev_close: float = None) -> list:
    '''
    Computes returns: r_t = (p_t - p_{t-1}) / p_{t-1}
    Returns list of tuples: (timestamp_utc, close, return)
    If prev_close is provided, computes return for the first observation against prev_close.
    Otherwise, the first observation has return = None.
    '''
    out = []
    for i, (dt, close_val) in enumerate(rows):
        if i == 0:
            if prev_close is not None and prev_close > 0:
                ret = (close_val - prev_close) / prev_close
            else:
                ret = None
        else:
            p_close = rows[i - 1][1]
            ret = (close_val - p_close) / p_close if p_close > 0 else 0.0
        out.append((dt, close_val, ret))
    return out


def load_existing_stock_returns(target_csv: Path) -> tuple[list, datetime | None, float | None]:
    '''
    Loads existing stock returns CSV file.
    Returns:
      (existing_rows, latest_timestamp_utc, latest_close_price)
    '''
    if not target_csv.exists():
        return [], None, None

    rows = []
    latest_ts = None
    latest_close = None

    with open(target_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp_utc", "").strip()
            close_str = row.get("close", "").strip()
            ret_str = row.get("return", "").strip()

            if not ts_str or not close_str:
                continue

            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            try:
                close_val = float(close_str)
            except ValueError:
                continue

            ret_val = float(ret_str) if ret_str else None
            rows.append((dt, close_val, ret_val))

    if rows:
        rows.sort(key=lambda x: x[0])
        latest_ts = rows[-1][0]
        latest_close = rows[-1][1]

    return rows, latest_ts, latest_close


def save_stock_returns_to_csv(target_csv: Path, rows: list) -> int:
    '''
    Saves stock price and stock return rows into a CSV file with columns:
    timestamp_utc, close, return.
    '''
    if not rows:
        log.warning(f"No rows to save for {target_csv.name}.")
        return 0

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "close", "return"])
        for dt, close_val, ret_val in rows:
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
            ret_str = f"{ret_val:.10f}" if ret_val is not None else ""
            writer.writerow([ts_str, close_val, ret_str])

    return len(rows)


def update_stock_series(target_csv: Path, ticker: str = "PYPL", current_ts: datetime = None) -> int:
    '''
    Updates stock returns incrementally: fetches via NASDAQ API only if latest timestamp is not today.
    '''
    if current_ts is None:
        current_ts = datetime.now(timezone.utc)

    existing_rows, latest_ts, last_close = load_existing_stock_returns(target_csv)

    if existing_rows and latest_ts:
        log.info(f"[{ticker}] Found existing dataset with {len(existing_rows)} records up to {latest_ts.strftime('%Y-%m-%d')}.")
        
        # Only fetch from NASDAQ API if latest timestamp is not today
        if latest_ts.date() == current_ts.date():
            log.info(f"[{ticker}] Latest timestamp ({latest_ts.strftime('%Y-%m-%d')}) is already today ({current_ts.strftime('%Y-%m-%d')}). Skipping NASDAQ API fetch.")
            return len(existing_rows)

        log.info(f"[{ticker}] Latest timestamp ({latest_ts.strftime('%Y-%m-%d')}) is not today ({current_ts.strftime('%Y-%m-%d')}). Fetching from NASDAQ API...")
        new_price_rows = fetch_stock_prices_from_nasdaq_api(ticker, min_ts=latest_ts, max_ts=current_ts)

        if new_price_rows:
            log.info(f"[{ticker}] Found {len(new_price_rows)} new price records. Computing returns from last close ({last_close:.2f})...")
            new_return_rows = compute_stock_returns(new_price_rows, prev_close=last_close)
            
            # Combine and deduplicate
            seen_dates = {r[0].strftime("%Y-%m-%d"): r for r in existing_rows}
            for dt, c_val, r_val in new_return_rows:
                seen_dates[dt.strftime("%Y-%m-%d")] = (dt, c_val, r_val)
            
            combined_rows = sorted(seen_dates.values(), key=lambda x: x[0])
            n = save_stock_returns_to_csv(target_csv, combined_rows)
            log.info(f"[{ticker}] Added {len(new_return_rows)} new records. Saved total of {n} records into {target_csv.name}.")
            return n
        else:
            log.info(f"[{ticker}] Dataset in {target_csv.name} is already up to date. No new records found.")
            return len(existing_rows)
    else:
        log.info(f"[{ticker}] No existing dataset found. Fetching full history from NASDAQ API up to {current_ts.strftime('%Y-%m-%d')} ...")
        price_rows = fetch_stock_prices_from_nasdaq_api(ticker, max_ts=current_ts)
        log.info(f"[{ticker}] Parsed {len(price_rows)} price records.")
        return_rows = compute_stock_returns(price_rows)
        n = save_stock_returns_to_csv(target_csv, return_rows)
        log.info(f"[{ticker}] Saved {n} records into {target_csv.name}.")
        return n


def run():
    current_ts = datetime.now(timezone.utc)
    log.info(f"Running stock returns collector (current timestamp: {current_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}) ...")

    # PYPL historical returns
    update_stock_series(DATA_DIR / "pypl_returns.csv", "PYPL", current_ts)

    log.info("Stock returns data collection completed.")


if __name__ == "__main__":
    run()
