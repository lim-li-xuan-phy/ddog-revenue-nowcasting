'''
run_all_collectors.py
=====================
Master orchestrator script that runs all data collection scripts:
  1. collect_stock_returns.py  (pypl_returns.csv)
  2. collect_tech_headlines.py (tech_headlines.csv with NLP news_score in [-1.0, 1.0])
  3. collect_announcements.py  (announcements.csv with product_release classifications)
  4. collect_qtrly_metrics.py  (qtrly_metrics.csv)

Incremental updating: collectors only fetch and process delta records between the latest
timestamp in existing CSV datasets and the current timestamp.
'''

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import collect_stock_returns
import collect_tech_headlines
import collect_announcements
import collect_qtrly_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CODE_DIR = Path(__file__).parent.resolve()
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"


def verify_csv_counts():
    """
    Reads CSV files in Data directory to output total row counts and date ranges.
    """
    log.info("Verifying CSV data files row counts and timestamps...")

    files = [
        "pypl_returns.csv",
        "tech_headlines.csv",
        "announcements.csv",
        "qtrly_metrics.csv",
    ]

    print("\n" + "=" * 78)
    print(f"{'File Name':<20} {'Row Count':>10} {'Min Timestamp (UTC)':>24} {'Max Timestamp (UTC)':>24}")
    print("=" * 78)
    for fname in files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"{fname:<20} {'MISSING':>10} {'N/A':>24} {'N/A':>24}")
            continue

        timestamps = []
        count = 0
        try:
            with open(fpath, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    count += 1
                    ts_str = row.get("timestamp_utc", "").strip()
                    if ts_str:
                        timestamps.append(ts_str)

            min_ts = min(timestamps)[:19] + " UTC" if timestamps else "N/A"
            max_ts = max(timestamps)[:19] + " UTC" if timestamps else "N/A"
            print(f"{fname:<20} {count:>10} {min_ts:>24} {max_ts:>24}")
        except Exception as e:
            print(f"{fname:<20} {'ERROR':>10} {str(e):>49}")
    print("=" * 78 + "\n")


def main():
    log.info("Starting master data collection pipeline (up to Q2 2026)...")

    # Stock Returns (DDOG & PYPL)
    log.info("\n--- [Source 1 & 2] Stock Returns Collector ---")
    collect_stock_returns.run()

    # Tech Headlines & NLP Sentiment
    log.info("\n--- [Source 3] Tech Headlines Collector ---")
    collect_tech_headlines.run()

    # Datadog Press Releases
    log.info("\n--- [Source 4] Datadog Announcements Collector ---")
    collect_announcements.run()

    # Quarterly Financial & Operating Metrics
    log.info("\n--- SEC EDGAR Quarterly Metrics Collector ---")
    collect_qtrly_metrics.main()

    # Summary Check
    verify_csv_counts()

    log.info("Data collection pipeline executed successfully.")


if __name__ == "__main__":
    main()
