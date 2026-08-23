'''
Scrapes Datadog (NASDAQ: DDOG) key financial and operational metrics from the
SEC EDGAR API and stores the results in `qtrly_metrics.csv` in DATA_DIR.

Metrics collected per quarter:
  - Revenue (USD)
  - Revenue QoQ Growth (%)
  - Revenue YoY Growth (%)
  - Remaining Performance Obligations (RPO) (USD)
  - RPO QoQ Growth (%)
  - RPO YoY Growth (%)
  - Calculated Billings (USD)  = Revenue + Change in Deferred Revenue
  - Billings QoQ Growth (%)
  - Billings YoY Growth (%)
  - Large Customer Count (ARR >= $100k)
  - Large Customer QoQ Growth (%)
  - Large Customer YoY Growth (%)
  - Net Revenue Retention (%)

Data Sources:
  - SEC EDGAR Company Facts API  (structured XBRL data)
  - SEC EDGAR Submissions API    (filing index & URLs)
  - SEC EDGAR Filing HTML        (text scraping for customer count & net retention)

Usage:
  python collect_qtrly_metrics.py [--refresh] [--loglevel {DEBUG,INFO,WARNING,ERROR}]

  --refresh   Force re-download of all HTML filings (ignore local cache).
              Without this flag the script re-uses cached HTML files.
'''

import argparse
import csv
import os
import re
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Base paths
CODE_DIR = Path(__file__).parent.resolve()
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DDOG_CIK          = "0001561550"
DDOG_CIK_PLAIN    = "1561550"           # without leading zeros
COMPANY_FACTS_URL = (
    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{DDOG_CIK}.json"
)
SUBMISSIONS_URL   = (
    f"https://data.sec.gov/submissions/CIK{DDOG_CIK}.json"
)
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Local cache directory
CACHE_DIR = Path(__file__).parent / ".sec_cache"

# Minimum inter-request delay to stay well under 10 req/s SEC EDGAR API limit
REQUEST_DELAY_S = 0.15

# Map each quarter to respective dates
Q_END_MONTH = {"Q1": "-03-31", "Q2": "-06-30", "Q3": "-09-30", "Q4": "-12-31"}
MONTH_TO_Q  = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}

# ---------------------------------------------------------------------------
# Set up environment
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """
    Load SEC EDGAR user-agent header from .env file.
    """
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    sec_header = os.getenv("SEC_EDGAR_HEADER")
    if not sec_header:
        log.error(
            "Environment variable 'SEC_EDGAR_HEADER' is not set. "
            "Add it to Code/.env before running."
        )
        sys.exit(1)

    return {"sec_header": sec_header}

# ---------------------------------------------------------------------------
# SEC EDGAR HTTP object
# ---------------------------------------------------------------------------
class SecClient:
    '''
    Wrapper functions to handle HTTP requests to SEC EDGAR API.
    '''

    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":      user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept":          "application/json, text/html, */*",
        })
        self._last_request_time = 0.0

    def _throttle(self):
        '''
        Limits the execution of a function to have REQUEST_DELAY_S time elapsed between calls.
        '''
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S - elapsed)

    def get_json(self, url: str) -> dict:
        self._throttle()
        resp = self.session.get(url, timeout=30)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def get_html(self, url: str) -> str:
        self._throttle()
        resp = self.session.get(url, timeout=60)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp.text

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _duration_days(item: dict) -> int:
    '''
    Returns the duration in days for a duration-type XBRL item.
    '''
    s, e = item.get("start"), item.get("end")
    if s and e:
        return (
            datetime.strptime(e, "%Y-%m-%d") - datetime.strptime(s, "%Y-%m-%d")
        ).days + 1
    return 0


def find_data(
    items: list,
    end_date: str,
    days_min: int = None,
    days_max: int = None,
) -> dict:
    '''
    Returns the best-matching XBRL item for a given end date of a quarter and optional 
    duration filter.

    Order of preference:
      1. The entry that has a `frame` value (assigned by SEC EDGAR as the closest fit to the 
      calendrical period requested)
      2. The entry filed latest
    '''
    candidates = []
    for item in items:
        if not item.get("end", "").startswith(end_date):
            continue
        if days_min is not None or days_max is not None:
            d = _duration_days(item)
            if days_min is not None and d < days_min:
                continue
            if days_max is not None and d > days_max:
                continue
        candidates.append(item)

    if not candidates:
        return None

    # Prefer the candidate with `frame`, otherwise filed latest.
    candidates.sort(key=lambda x: (bool(x.get("frame")), x.get("filed", "")))
    return candidates[-1]


def extract_xbrl_metrics(facts_json: dict) -> dict:
    '''
    Parse the JSON response from SEC EDGAR company facts API and returns a dict keyed by 
    "YYYY-Q#" (e.g. "2024-Q3") containing financial data.

    -Returns-
    { "YYYY-Q#": {"revenue": int|None, "deferred_revenue": int|None, "rpo": int|None, 
                    "billings": int|None} }
    '''
    us_gaap = facts_json["facts"]["us-gaap"]

    def usd_items(tag: str) -> list:
        return us_gaap.get(tag, {}).get("units", {}).get("USD", [])

    rev_items          = usd_items("RevenueFromContractWithCustomerExcludingAssessedTax") # exclude tax bc the amount does not belong to the company
    def_curr_items     = usd_items("ContractWithCustomerLiabilityCurrent")
    def_noncurr_items  = usd_items("ContractWithCustomerLiabilityNoncurrent")
    # Fallback for very early periods that used legacy GAAP tags
    def_curr_legacy    = usd_items("DeferredRevenueCurrent")
    def_noncurr_legacy = usd_items("DeferredRevenueNoncurrent")
    rpo_items          = usd_items("RevenueRemainingPerformanceObligation")

    periods = {}

    current_year = datetime.now().year

    for year in range(2019, current_year + 1):
        q_specs = [
            ("Q1", f"{year}-03", 80,   100),
            ("Q2", f"{year}-06", 80,   100),
            ("Q3", f"{year}-09", 80,   100),
            ("Q4", f"{year}-12", None, None),   # calculated below bc companies don't report Q4 by itself
        ]

        for q_name, date_pfx, d_min, d_max in q_specs:
            key = f"{year}-{q_name}"
            if key not in periods:
                periods[key] = {
                    "revenue":          None,
                    "deferred_revenue": None,
                    "rpo":              None,
                    "billings":         None,
                }

            # Revenue
            if q_name != "Q4":
                item = find_data(rev_items, date_pfx, d_min, d_max)
                if item:
                    periods[key]["revenue"] = item["val"]
            else:
                # Q4 revenue = Full-year revenue minus Q3 YTD revenue
                fy_item  = find_data(rev_items, f"{year}-12", 350, 380)
                ytd_item = find_data(rev_items, f"{year}-09", 250, 290)
                if fy_item and ytd_item:
                    periods[key]["revenue"] = fy_item["val"] - ytd_item["val"]

            # Point-in-time Deferred Revenue
            curr_item = (
                find_data(def_curr_items, date_pfx)
                or find_data(def_curr_legacy, date_pfx)
            )
            noncurr_item = (
                find_data(def_noncurr_items, date_pfx)
                or find_data(def_noncurr_legacy, date_pfx)
            )
            curr_val    = curr_item["val"]    if curr_item    else 0
            noncurr_val = noncurr_item["val"] if noncurr_item else 0
            if curr_val > 0 or noncurr_val > 0:
                periods[key]["deferred_revenue"] = curr_val + noncurr_val

            # RPO
            rpo_item = find_data(rpo_items, date_pfx)
            if rpo_item:
                periods[key]["rpo"] = rpo_item["val"]

    # Compute billings
    # Billings_t = Revenue_t + (DeferredRevenue_t - DeferredRevenue_{t-1})
    sorted_keys = sorted( 
        periods.keys(),
        key=lambda k: (int(k[:4]), {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}[k[5:]]),
    ) # sorting quarters chronologically
    for i in range(1, len(sorted_keys)):
        key  = sorted_keys[i]
        prev = sorted_keys[i - 1]
        rev      = periods[key]["revenue"]
        def_now  = periods[key]["deferred_revenue"]
        def_prev = periods[prev]["deferred_revenue"]
        if rev is not None and def_now is not None and def_prev is not None:
            periods[key]["billings"] = rev + (def_now - def_prev)

    return periods


def _clean_html_text(html: str) -> str:
    '''
    Removes HTML tags and whitespace.
    '''
    soup = BeautifulSoup(html, "html.parser")
    raw  = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", raw)


def extract_operational_metrics(text: str) -> dict:
    """
    Extracts operational metrics from filing plain text:
      - `large_customer_count`: customers with ARR >= $100k
      - `net_revenue_retention`: dollar-based net retention rate parsed according to rules:
          - If phrase contains 'low', last digit is 2 (e.g. low-120%'s -> 122%)
          - If phrase contains 'about', last digit is 0 (e.g. about 120% -> 120%)
          - If phrase contains 'mid', last digit is 5 (e.g. mid-110%'s -> 115%)
          - If phrase contains 'high', last digit is 8 (e.g. high-110%'s -> 118%)
          - If phrase contains 'above', last digit is 1 (e.g. above 130% -> 131%)
      - `rpo`: Remaining performance obligations in USD (fallback when omitted in XBRL facts)
    """
    result = {
        "large_customer_count": None,
        "net_revenue_retention": None,
        "rpo": None,
    }

    # Large customer count ($100k+ ARR)
    m_cust = re.search(
        r"(?i)(?:we\s+had|had)\s+(?:approximately\s+|about\s+)?([\d,]+)\s+customers\s+(?:with|who\s+had)\s+[^.]+?\$100,000",
        text,
    )
    if not m_cust:
        m_cust = re.search(
            r"(?i)(?:approximately|about)\s+([\d,]+)\s+customers\s+(?:with|who\s+had)\s+[^.]+?\$100,000",
            text,
        )
    if not m_cust:
        m_cust = re.search(
            r"(?i)([\d,]+)\s+customers\s+with\s+(?:annual\s+run-rate\s+revenue|annual\s+recurring\s+revenue|an?\s+ARR|ARR)[^.]+?\$100,000",
            text,
        )
    if m_cust:
        result["large_customer_count"] = int(m_cust.group(1).replace(",", ""))

    # Dollar-based net retention rate or Net revenue retention
    m_nrr = re.search(
        r"dollar-based\s+net\s+retention\s+rate\s+was\s+([^.]+)",
        text,
        re.IGNORECASE,
    )
    if m_nrr:
        phrase = m_nrr.group(1).strip().lower()
        num_match = re.search(r"(\d{2,3})", phrase)
        if num_match:
            base_num = int(num_match.group(1))
            tens = (base_num // 10) * 10

            has_low = bool(re.search(r"\blow\b", phrase))
            has_about = bool(re.search(r"\babout\b|\bapproximately\b", phrase))
            has_mid = bool(re.search(r"\bmid\b", phrase))
            has_high = bool(re.search(r"\bhigh\b", phrase))
            has_above = bool(re.search(r"\babove\b", phrase))

            if has_low:
                val = tens + 2
            elif has_mid:
                val = tens + 5
            elif has_high:
                val = tens + 8
            elif has_about:
                val = tens + 0
            elif has_above:
                val = tens + 1
            else:
                val = base_num

            result["net_revenue_retention"] = float(val)

    # RPO fallback from filing text (e.g. 2021 10-Qs where XBRL tag had dimensional segment)
    m_rpo = re.search(
        r"allocated\s+to\s+remaining\s+performance\s+obligations\s+was\s+\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?",
        text,
        re.IGNORECASE,
    )
    if not m_rpo:
        m_rpo = re.search(
            r"remaining\s+performance\s+obligations[^\.\$]{0,100}?(?:was|were|is|of)\s+\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?",
            text,
            re.IGNORECASE,
        )
    if m_rpo:
        val_str = m_rpo.group(1).replace(",", "")
        unit = (m_rpo.group(2) or "").lower()
        multiplier = 1
        if unit == "million":
            multiplier = 1_000_000
        elif unit == "billion":
            multiplier = 1_000_000_000
        result["rpo"] = int(float(val_str) * multiplier)

    return result


def get_periodic_filings(sub_json: dict, client: SecClient = None) -> list:
    '''
    Gets list of 10-Q (Q1-Q3 reports) and 10-K (annual reports) filing metadata, sorted by 
    report date. Fetches older submission files if client is provided.
    '''
    containers = [sub_json["filings"]["recent"]]
    if client:
        for file_meta in sub_json["filings"].get("files", []):
            f_url = f"https://data.sec.gov/submissions/{file_meta['name']}"
            try:
                containers.append(client.get_json(f_url))
            except Exception as e:
                log.warning(f"Could not fetch older submissions file {file_meta['name']}: {e}")

    filings = []
    seen_accessions = set()
    for container in containers:
        forms = container["form"]
        for i, form in enumerate(forms):
            if form not in ("10-Q", "10-K"):
                continue
            acc = container["accessionNumber"][i]
            if acc in seen_accessions:
                continue
            seen_accessions.add(acc)
            filings.append({
                "form":        form,
                "accession":   acc,
                "report_date": container["reportDate"][i],
                "filing_date": container["filingDate"][i],
                "primary_doc": container["primaryDocument"][i],
            })
    filings.sort(key=lambda x: x["report_date"])
    return filings


def get_filing_text(
    client: SecClient,
    accession: str,
    primary_doc: str,
    refresh: bool,
) -> str:
    """
    Returns the plain text content of a filing using a local cache when available. HTML files are cached 
    in the .sec_cache/ folder.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    acc_clean  = accession.replace("-", "")
    cache_file = CACHE_DIR / f"{acc_clean}_{primary_doc}"

    if not refresh and cache_file.exists():
        log.debug(f"Cache hit: {cache_file.name}")
        return _clean_html_text(cache_file.read_text(encoding="utf-8"))

    url  = f"{SEC_ARCHIVES_BASE}/{DDOG_CIK_PLAIN}/{acc_clean}/{primary_doc}"
    log.info(f"  Downloading {url}")
    html = client.get_html(url)
    cache_file.write_text(html, encoding="utf-8")
    return _clean_html_text(html)


def load_existing_qtrly_metrics(target_csv: Path) -> tuple[dict, str | None, datetime | None]:
    '''
    Loads existing quarterly metrics from target_csv.
    Returns (dict_of_rows, latest_period_end, latest_timestamp).
    '''
    if not target_csv.exists():
        return {}, None, None

    records = {}
    latest_period_end = None
    latest_ts = None

    try:
        with open(target_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp_utc", "").strip()
                q_label = row.get("quarter_label", "").strip()
                if not ts_str or not q_label:
                    continue
                try:
                    dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                period_end = ts_str[:10]
                records[q_label] = row
                records[q_label.replace(" ", "-")] = row
                if latest_ts is None or dt > latest_ts:
                    latest_ts = dt
                    latest_period_end = period_end
    except Exception as e:
        log.warning(f"Could not read existing file {target_csv.name}: {e}")
        return {}, None, None

    return records, latest_period_end, latest_ts


def build_quarterly_records(
    xbrl_periods: dict,
    filings: list,
    client: SecClient,
    refresh: bool,
    latest_period_end: str = None,
) -> list:
    """
    xbrl_periods: XBRL financial data from extract_xbrl_metrics().
    filings: Filing metadata of all 10-K and 10-Q reports from get_periodic_filings().

    Merges XBRL financial data and operational metrics and computes QoQ and YoY growth rates.
    Returns a list of record dicts, sorted chronologically.
    """
    # Scrape each 10-Q and 10-K filing for financial data of interest
    op_metrics = {}
    for filing in filings:
        rd     = filing["report_date"]
        year   = rd[:4]
        month  = rd[5:7]
        q_name = MONTH_TO_Q.get(month)
        if q_name is None:
            continue
        key = f"{year}-{q_name}"

        # If incremental mode and filing is older than latest_period_end, we can still load from cache or skip if not refresh
        log.info(f"Parsing {filing['form']} {rd}  ({key})")
        text = get_filing_text(
            client, filing["accession"], filing["primary_doc"], refresh
        )
        op_metrics[key] = extract_operational_metrics(text)
        op_metrics[key]["form"] = filing["form"]

    # Sort XBRL financial data chronologically
    sorted_keys = sorted(
        xbrl_periods.keys(),
        key=lambda k: (int(k[:4]), {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}[k[5:]]),
    )

    records = []
    for key in sorted_keys:
        year, q_name = key[:4], key[5:]
        period_end   = year + Q_END_MONTH[q_name]
        xbrl = xbrl_periods[key]
        op   = op_metrics.get(key, {})

        # RPO: prefer XBRL, fallback to filing text if XBRL omitted dimensional fact
        rpo_val = xbrl.get("rpo")
        if rpo_val is None:
            rpo_val = op.get("rpo")

        records.append({
            "period_end":            period_end,
            "fiscal_year":           int(year),
            "fiscal_quarter":        q_name,
            "form":                  op.get("form"),
            "revenue":               xbrl.get("revenue"),
            "revenue_qoq":           None,
            "revenue_yoy":           None,
            "rpo":                   rpo_val,
            "rpo_qoq":               None,
            "rpo_yoy":               None,
            "billings":              xbrl.get("billings"),
            "billings_qoq":          None,
            "billings_yoy":          None,
            "large_customer_count":  op.get("large_customer_count"),
            "large_customer_qoq":    None,
            "large_customer_yoy":    None,
            "net_revenue_retention": op.get("net_revenue_retention"),
            "deferred_revenue":      xbrl.get("deferred_revenue"),
        })

    def calc_growth(curr, old):
        '''
        Calculates growth rate from an old datapoint to the current datapoint.
        '''
        if curr is not None and old is not None and old != 0:
            return round(((curr / old) - 1) * 100, 4)
        return None

    for i, rec in enumerate(records):
        # QoQ growth: compare with record 1 quarter prior
        if i >= 1:
            prev_qoq = records[i - 1]
            rec["revenue_qoq"]        = calc_growth(rec["revenue"],              prev_qoq["revenue"])
            rec["rpo_qoq"]            = calc_growth(rec["rpo"],                  prev_qoq["rpo"])
            rec["billings_qoq"]       = calc_growth(rec["billings"],             prev_qoq["billings"])
            rec["large_customer_qoq"] = calc_growth(rec["large_customer_count"], prev_qoq["large_customer_count"])

        # YoY growth: compare with record 4 quarters prior (1 year prior)
        if i >= 4:
            prev_yoy = records[i - 4]
            rec["revenue_yoy"]        = calc_growth(rec["revenue"],              prev_yoy["revenue"])
            rec["rpo_yoy"]            = calc_growth(rec["rpo"],                  prev_yoy["rpo"])
            rec["billings_yoy"]       = calc_growth(rec["billings"],             prev_yoy["billings"])
            rec["large_customer_yoy"] = calc_growth(rec["large_customer_count"], prev_yoy["large_customer_count"])

    return records


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    # (record_key,            column_name)
    ("revenue",               "revenue (USD)"),
    ("revenue_qoq",           "revenue_qoq (%)"),
    ("revenue_yoy",           "revenue_yoy (%)"),
    ("rpo",                   "rpo (USD)"),
    ("rpo_qoq",               "rpo_qoq (%)"),
    ("rpo_yoy",               "rpo_yoy (%)"),
    ("billings",              "billings (USD)"),
    ("billings_qoq",          "billings_qoq (%)"),
    ("billings_yoy",          "billings_yoy (%)"),
    ("large_customer_count",  "large_customer_count"),
    ("large_customer_qoq",    "large_customer_qoq (%)"),
    ("large_customer_yoy",    "large_customer_yoy (%)"),
    ("net_revenue_retention", "net_revenue_retention (%)"),
]


def save_qtrly_metrics_to_csv(target_csv: Path, records: list) -> int:
    '''
    records: List of record dicts. Each dict stores the information of a quarter.

    Saves quarterly metrics into a CSV file with columns in the specified order:
      timestamp_utc, quarter_label, Revenue, Revenue QoQ, Revenue YoY,
      RPO, RPO QoQ, RPO YoY, Billings, Billings QoQ, Billings YoY,
      Large Customer Count, Large Customer QoQ, Large Customer YoY, Net Revenue Retention.
    Returns the number of rows written.
    '''
    visible_records = [r for r in records if r.get("revenue") is not None]
    if not visible_records:
        log.warning("No metric rows to save.")
        return 0

    header = ["timestamp_utc", "quarter_label"] + [col_name for _, col_name in METRIC_COLUMNS]

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in visible_records:
                period_end = r.get("period_end")
                if not period_end:
                    continue
                ts = datetime.strptime(period_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S+00:00")
                quarter_label = f"{r.get('fiscal_year')} {r.get('fiscal_quarter')}"

                row = [ts_str, quarter_label]
                for key, _ in METRIC_COLUMNS:
                    val = r.get(key)
                    row.append(val if val is not None else "")
                writer.writerow(row)
    except PermissionError:
        log.error(
            f"Permission denied when writing to '{target_csv.resolve()}'. "
            "Please ensure the file is closed in Excel or other programs and re-run."
        )
        raise

    return len(visible_records)

def print_table(records: list) -> None:
    '''
    Prints a table of the quarterly metrics to console.
    '''
    visible = [r for r in records if r["revenue"] is not None]

    def fmt_usd(v, scale=1e6) -> str:
        return f"${v/scale:8.1f}M" if v is not None else "     N/A"

    def fmt_pct(v) -> str:
        return f"{v:+6.1f}%" if v is not None else "   N/A"

    def fmt_nrr(v) -> str:
        return f"{v:5.0f}%" if v is not None else "  N/A"

    def fmt_int(v) -> str:
        return f"{v:6,}" if v is not None else "   N/A"

    header = (
        f"{'Period':<11} {'Form':<5} "
        f"{'Revenue':>9} {'RevQoQ':>7} {'RevYoY':>7} "
        f"{'RPO':>9} {'RPOQoQ':>7} {'RPOYoY':>7} "
        f"{'Billings':>9} {'BillQoQ':>8} {'BillYoY':>8} "
        f"{'#Cust':>7} {'CustQoQ':>8} {'CustYoY':>8} "
        f"{'NRR':>5}"
    )
    sep = "-" * len(header)
    print()
    print("  Datadog (DDOG) Quarterly Metrics scraped from SEC EDGAR API")
    print(sep)
    print(header)
    print(sep)
    for r in visible:
        print(
            f"{r['period_end']:<11} "
            f"{(r['form'] or ''):5} "
            f"{fmt_usd(r['revenue']):>9} "
            f"{fmt_pct(r['revenue_qoq']):>7} "
            f"{fmt_pct(r['revenue_yoy']):>7} "
            f"{fmt_usd(r['rpo']):>9} "
            f"{fmt_pct(r['rpo_qoq']):>7} "
            f"{fmt_pct(r['rpo_yoy']):>7} "
            f"{fmt_usd(r['billings']):>9} "
            f"{fmt_pct(r['billings_qoq']):>8} "
            f"{fmt_pct(r['billings_yoy']):>8} "
            f"{fmt_int(r['large_customer_count']):>7} "
            f"{fmt_pct(r['large_customer_qoq']):>8} "
            f"{fmt_pct(r['large_customer_yoy']):>8} "
            f"{fmt_nrr(r['net_revenue_retention']):>5}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Datadog SEC EDGAR metrics and store in CSV."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of all HTML filings (ignore local cache).",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.loglevel))

    target_csv = DATA_DIR / "qtrly_metrics.csv"
    current_ts = datetime.now(timezone.utc)
    existing_data, latest_period_end, latest_ts = load_existing_qtrly_metrics(target_csv)

    # Set up environment
    cfg    = load_config()
    client = SecClient(user_agent=cfg["sec_header"])

    # Fetch XBRL data
    log.info("Fetching Datadog XBRL company facts from SEC EDGAR ...")
    facts_json   = client.get_json(COMPANY_FACTS_URL)
    log.info("Extracting quarterly XBRL metrics ...")
    xbrl_periods = extract_xbrl_metrics(facts_json)

    # Fetch filing index
    log.info("Fetching filing submission index ...")
    sub_json = client.get_json(SUBMISSIONS_URL)
    filings  = get_periodic_filings(sub_json, client)
    log.info(f"Found {len(filings)} periodic filings (10-Q / 10-K).")

    if not args.refresh and existing_data and latest_period_end:
        log.info(f"Found existing dataset with quarterly metrics up to {latest_period_end}.")
        # Check if there are any filings or XBRL quarters newer than latest_period_end
        new_filings = [f for f in filings if f["report_date"] > latest_period_end]
        new_xbrl_periods = {
            k: v for k, v in xbrl_periods.items()
            if v.get("revenue") is not None and k not in existing_data and k.replace("-", " ") not in existing_data
        }

        if not new_filings and not new_xbrl_periods:
            log.info(f"Quarterly metrics in {target_csv.name} are already up to date (latest: {latest_period_end}). No new quarters reported.")
            return

    # Build complete quarterly records
    log.info("Downloading / parsing filing HTML for operational metrics ...")
    records = build_quarterly_records(xbrl_periods, filings, client, args.refresh, latest_period_end)

    # Print to console
    print_table(records)

    # Save data to CSV
    n = save_qtrly_metrics_to_csv(target_csv, records)
    log.info(f"Saved {n} metric rows into {target_csv.name}.")

    log.info("Quarterly metrics collection completed.")


if __name__ == "__main__":
    main()
