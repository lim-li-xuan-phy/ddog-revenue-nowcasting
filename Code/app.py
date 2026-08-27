import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import sys
import logging
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")

# --- BASE PATHS ---
CODE_DIR = Path(__file__).parent.resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
ASSETS_DIR = BASE_DIR / "Assets"

def run_data_collectors():
    """
    Executes data collection and backtesting scripts to update
    data and results CSV files.
    """
    updated_files = []
    
    # Stock returns data
    try:
        import collect_stock_returns
        collect_stock_returns.run()
        updated_files.append("pypl_returns.csv")
    except Exception as e:
        log.warning(f"Error running collect_stock_returns: {e}")

    # Product releases data
    try:
        import collect_announcements
        collect_announcements.run()
        updated_files.append("announcements.csv")
    except Exception as e:
        log.warning(f"Error running collect_announcements: {e}")

    # Quarterly metrics reported by Datadog
    try:
        import collect_qtrly_metrics
        collect_qtrly_metrics.main()
        updated_files.append("qtrly_metrics.csv")
    except Exception as e:
        log.warning(f"Error running collect_qtrly_metrics: {e}")

    # Out-of-sample walk-forward backtesting
    try:
        import run_backtest
        df_bt, latest_bt_q, detailed_records = run_backtest.run()
        updated_files.append("backtesting_summary.csv")
        updated_files.append("every_backtest_prediction.csv")
        updated_files.append("walkforward_backtest.png")
    except Exception as e:
        log.warning(f"Error running walk-forward backtesting: {e}")

    return updated_files


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Datadog Revenue Nowcast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS ---
st.markdown("""
<style>
    /* Uncrop sidebar logo */
    [data-testid="stSidebar"] img, 
    [data-testid="stSidebarUserContent"] img, 
    .stImage img,
    div[data-testid="stImage"] img {
        border-radius: 0px !important;
        clip-path: none !important;
        mask: none !important;
        -webkit-mask: none !important;
        object-fit: contain !important;
        background: transparent !important;
    }
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #475569;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        min-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-title {
        font-size: 11.5px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
        margin: 2px 0 6px 0;
    }
    .metric-footer {
        font-size: 12px;
        font-weight: 600;
    }
    .badge-beat {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12.5px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-miss {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12.5px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-inline {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12.5px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_quarterly_data():
    qtr_path = DATA_DIR / "qtrly_metrics.csv"
    if qtr_path.exists():
        df = pd.read_csv(qtr_path)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        return df.sort_values('timestamp_utc').reset_index(drop=True)
    return None

@st.cache_data
def load_backtest_summary():
    bt_path = RESULTS_DIR / "Summaries" / "backtesting_summary.csv"
    if bt_path.exists():
        return pd.read_csv(bt_path)
    return None

@st.cache_data
def load_backtest_detailed():
    detailed_path = RESULTS_DIR / "Summaries" / "every_backtest_prediction.csv"
    if detailed_path.exists():
        df = pd.read_csv(detailed_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    try:
        import run_backtest
        df_summary, latest_q, detailed_records = run_backtest.run()
        if detailed_path.exists():
            df = pd.read_csv(detailed_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        log.warning(f"Error loading or generating backtest detailed records: {e}")
    return None

@st.cache_data
def load_correlations():
    corr_path = RESULTS_DIR / "Summaries" / "strongest_correlations.csv"
    if corr_path.exists():
        return pd.read_csv(corr_path)
    return None

@st.cache_data
def load_raw_signals():
    """
    Loads raw alternative datasets for empirical rolling calculations.
    """
    pypl_path = DATA_DIR / "pypl_returns.csv"
    tech_path = DATA_DIR / "tech_headlines.csv"
    ann_path = DATA_DIR / "announcements.csv"

    s_pypl = pd.Series(dtype=float)
    if pypl_path.exists():
        s_pypl = pd.read_csv(
            pypl_path, parse_dates=['timestamp_utc']
        ).set_index('timestamp_utc')['return'].dropna().sort_index()

    s_tech = pd.Series(dtype=float)
    if tech_path.exists():
        s_tech = pd.read_csv(
            tech_path, parse_dates=['timestamp_utc']
        ).set_index('timestamp_utc')['news_score'].dropna().sort_index()

    s_ann = pd.Series(dtype=float)
    if ann_path.exists():
        df_ann = pd.read_csv(ann_path, parse_dates=['timestamp_utc'])
        if 'product_release' in df_ann.columns:
            is_rel = df_ann['product_release'].astype(str).str.strip().str.upper().isin(['TRUE', '1'])
            s_ann = df_ann[is_rel].set_index('timestamp_utc')['product_release'].sort_index()

    return s_pypl, s_tech, s_ann

def get_dataset_latency_metadata():
    """
    Reads CSV data and returns:
    Dataset, Source, Update Frequency, Latency, Freshness Status, and Coverage range.
    """
    now_utc = datetime.now(timezone.utc)

    csv_files = {
        "pypl_returns": {
            "name": "NASDAQ: PYPL Returns",
            "file": DATA_DIR / "pypl_returns.csv",
            "source": "NASDAQ Historical Data API",
            "freq": "Market daily (at close)",
            "latency": "T+0 End of Day",
            "latency_delta": timedelta(days=1),
            "coverage": "Since 12 Aug 2016",
            "date_col": "timestamp_utc",
        },
        "tech_headlines": {
            "name": "Tech Headlines",
            "file": DATA_DIR / "tech_headlines.csv",
            "source": "CNBC cloud technology news snapshots",
            "freq": "Hourly (Continuous)",
            "latency": "T+0 real-time",
            "latency_delta": timedelta(hours=24),
            "coverage": "Since 1 Jan 2021",
            "date_col": "timestamp_utc",
        },
        "announcements": {
            "name": "Product Releases",
            "file": DATA_DIR / "announcements.csv",
            "source": "Datadog Newsroom press releases",
            "freq": "Continuous",
            "latency": "Continuous",
            "latency_delta": timedelta(days=2),
            "coverage": "Since 20 Nov 2012",
            "date_col": "timestamp_utc",
        },
        "qtrly_metrics": {
            "name": "Financial & operating KPIs (DDOG)",
            "file": DATA_DIR / "qtrly_metrics.csv",
            "source": "SEC EDGAR Official Filings",
            "freq": "Quarterly",
            "latency": "T+40 Days Post-Quarter",
            "latency_delta": timedelta(days=50),
            "coverage": "Since 2019 Q1",
            "date_col": "timestamp_utc",
        },
    }

    metadata_rows = []
    for key, item in csv_files.items():
        fpath = item["file"]
        coverage_str = item["coverage"]
        
        if not fpath.exists():
            metadata_rows.append({
                "Dataset": item["name"],
                "Source": item["source"],
                "Update Frequency": item["freq"],
                "Latency": item["latency"],
                "Freshness Status": "🔴 Data not found",
                "Coverage": coverage_str,
            })
            continue

        try:
            df = pd.read_csv(fpath)
            date_col = item["date_col"]
            if date_col in df.columns and not df[date_col].dropna().empty:
                df[date_col] = pd.to_datetime(df[date_col])
                max_dt = df[date_col].max()

                if max_dt.tzinfo is None:
                    max_dt_cmp = max_dt.replace(tzinfo=timezone.utc)
                else:
                    max_dt_cmp = max_dt

                time_since_update = now_utc - max_dt_cmp
                days_ago = max(0, time_since_update.days)
                days_ago_str = f"{days_ago} day ago" if days_ago == 1 else f"{days_ago} days ago"

                if key == "qtrly_metrics":
                    latest_q = df['quarter_label'].iloc[-1] if 'quarter_label' in df.columns else ""
                    if time_since_update <= item["latency_delta"] or time_since_update <= timedelta(days=90):
                        freshness = f"🟢 Fresh (Complete up to {latest_q})"
                    else:
                        freshness = f"🟡 May need update (Last: {latest_q})"
                else:
                    if time_since_update < item["latency_delta"]:
                        freshness = f"🟢 Fresh ({days_ago_str})"
                    else:
                        freshness = f"🟡 May need update ({days_ago_str})"
            else:
                freshness = "🔴 Data not found"

            metadata_rows.append({
                "Dataset": item["name"],
                "Source": item["source"],
                "Update Frequency": item["freq"],
                "Latency": item["latency"],
                "Freshness Status": freshness,
                "Coverage": coverage_str,
            })
        except Exception:
            metadata_rows.append({
                "Dataset": item["name"],
                "Source": item["source"],
                "Update Frequency": item["freq"],
                "Latency": item["latency"],
                "Freshness Status": "🔴 Data not found",
                "Coverage": coverage_str,
            })

    return metadata_rows

# Load static/base dataframes
qtr_df = load_quarterly_data()
bt_df = load_backtest_summary()
bt_detailed_df = load_backtest_detailed()
corr_df = load_correlations()
s_pypl, s_tech, s_ann = load_raw_signals()

if qtr_df is None or qtr_df.empty:
    st.error("❌ Fatal Error: `qtrly_metrics.csv` is missing or empty. Please run the data collectors to ingest quarterly metrics.")
    st.stop()

if len(qtr_df) < 5:
    st.error(f"❌ Fatal Error: Insufficient quarterly data ({len(qtr_df)} quarters found). At least 5 historical quarters are required to compute YoY benchmarks.")
    st.stop()


# --- DYNAMIC QUARTER RESOLUTION ---
latest_row = qtr_df.iloc[-1]
prior_year_row = qtr_df.iloc[-4]
prior_quarter_prev_row = qtr_df.iloc[-2] if len(qtr_df) >= 2 else latest_row
prior_year_prev_row = qtr_df.iloc[-5] if len(qtr_df) >= 5 else prior_year_row

hist_rev = float(latest_row['revenue (USD)']) / 1e6
hist_rpo = float(latest_row['rpo (USD)']) / 1e6
hist_billings = float(latest_row['billings (USD)']) / 1e6
hist_customers = int(latest_row['large_customer_count'])

hist_rev_qoq = float(latest_row['revenue_qoq (%)'])
hist_rpo_yoy = float(latest_row['rpo_yoy (%)'])
hist_billings_yoy = float(latest_row['billings_yoy (%)'])
hist_cust_qoq = float(latest_row['large_customer_qoq (%)'])

prior_year_rev = float(prior_year_row['revenue (USD)']) / 1e6
prior_year_rpo = float(prior_year_row['rpo (USD)']) / 1e6
prior_year_billings = float(prior_year_row['billings (USD)']) / 1e6

hist_rev_prev = float(prior_quarter_prev_row['revenue (USD)']) / 1e6
prior_year_prev_rev = float(prior_year_prev_row['revenue (USD)']) / 1e6

# Parse latest quarter
latest_q_label = str(latest_row['quarter_label']).strip()
q_parts = latest_q_label.split()
cur_year = int(q_parts[0]) if len(q_parts) > 0 and q_parts[0].isdigit() else datetime.now().year
cur_q_num = int(q_parts[1].replace('Q', '')) if len(q_parts) > 1 and q_parts[1].replace('Q', '').isdigit() else 2

# Generate the next 4 quarter labels and quarter end dates
future_4_quarters = []
future_4_dates = []

for i in range(1, 5):
    tot_q = cur_q_num + i
    y = cur_year + (tot_q - 1) // 4
    q = ((tot_q - 1) % 4) + 1
    q_str = f"{y} Q{q}"
    future_4_quarters.append(q_str)
    
    end_month = q * 3
    end_day = 31 if end_month in [3, 12] else 30
    future_4_dates.append(pd.Timestamp(f"{y}-{end_month:02d}-{end_day:02d}", tz="UTC"))

# Target quarter for intra-quarter tracking is the immediate next quarter (Q+1)
target_quarter_label = future_4_quarters[0]
target_q_start_month = (int(target_quarter_label.split()[1].replace('Q', '')) - 1) * 3 + 1
target_q_year = int(target_quarter_label.split()[0])
target_q_start_date = pd.Timestamp(f"{target_q_year}-{target_q_start_month:02d}-01", tz="UTC")
target_q_end_date = future_4_dates[0]

target_dates = pd.date_range(target_q_start_date, target_q_end_date, freq='D')
total_days_in_quarter = len(target_dates)

# Real-world progress within target quarter
now_dt = datetime.now()
days_into_current_q = (now_dt - target_q_start_date.tz_localize(None)).days + 1
today_day_of_quarter = max(1, min(total_days_in_quarter, days_into_current_q))


# --- SIDEBAR CONTROLS ---
logo_path = ASSETS_DIR / "dd_logo_v_rgb.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=72)
else:
    st.sidebar.image("https://img.icons8.com/color/96/datadog.png", width=64)

st.sidebar.markdown("---")

# Target Fiscal Quarter Display
st.sidebar.markdown(f"**Current quarter: {target_quarter_label}**")

# Intra-quarter progress slider section
if "day_slider_input" not in st.session_state:
    st.session_state["day_slider_input"] = min(65, total_days_in_quarter)
elif st.session_state["day_slider_input"] > total_days_in_quarter:
    st.session_state["day_slider_input"] = total_days_in_quarter

col_d1, col_d2 = st.sidebar.columns([3, 2])
with col_d1:
    st.markdown("**📅 Day of quarter**")
with col_d2:
    if st.button("↺ Today", key="btn_reset_today"):
        st.session_state["day_slider_input"] = today_day_of_quarter
        st.rerun()

day_of_quarter = st.sidebar.slider(
    "Intra-quarter progress",
    min_value=1,
    max_value=total_days_in_quarter,
    step=1,
    key="day_slider_input",
    label_visibility="collapsed"
)

pct_complete = (day_of_quarter / float(total_days_in_quarter)) * 100
st.sidebar.markdown(f"**Quarter completion:** `{pct_complete:.1f}%` (Day {day_of_quarter}/{total_days_in_quarter})")
st.sidebar.markdown("---")


# --- SIDEBAR WEIGHT ADJUSTMENTS ---
st.sidebar.markdown("### ⚖️ Component weights")
st.sidebar.caption("Weighted contribution of each metric's momentum to the combined revenue nowcast:")

col_w_title, col_w_rst = st.sidebar.columns([3, 2])
with col_w_rst:
    if st.button("↺ Default", key="btn_reset_weights", help="Reset weights to 10% Rev, 30% RPO, 30% Billings, 30% Large Cust"):
        st.session_state["w_rev_slider"] = 10
        st.session_state["w_rpo_slider"] = 30
        st.session_state["w_bill_slider"] = 30
        st.session_state["w_cust_slider"] = 30
        st.rerun()

if "w_rev_slider" not in st.session_state:
    st.session_state["w_rev_slider"] = 10
if "w_rpo_slider" not in st.session_state:
    st.session_state["w_rpo_slider"] = 30
if "w_bill_slider" not in st.session_state:
    st.session_state["w_bill_slider"] = 30
if "w_cust_slider" not in st.session_state:
    st.session_state["w_cust_slider"] = 30

w_rev_raw = st.sidebar.slider("Revenue QoQ Growth", 0, 100, key="w_rev_slider", format="%d%%")
w_rpo_raw = st.sidebar.slider("RPO YoY Growth", 0, 100, key="w_rpo_slider", format="%d%%")
w_bill_raw = st.sidebar.slider("Billings YoY Growth", 0, 100, key="w_bill_slider", format="%d%%")
w_cust_raw = st.sidebar.slider("Large Customers QoQ", 0, 100, key="w_cust_slider", format="%d%%")

# Weight Normalization
total_raw_weight = w_rev_raw + w_rpo_raw + w_bill_raw + w_cust_raw
if total_raw_weight > 0:
    w_rev = w_rev_raw / total_raw_weight
    w_rpo = w_rpo_raw / total_raw_weight
    w_bill = w_bill_raw / total_raw_weight
    w_cust = w_cust_raw / total_raw_weight
else:
    w_rev, w_rpo, w_bill, w_cust = 0.10, 0.30, 0.30, 0.30

st.sidebar.caption(
    f"Normalized: Rev **{w_rev*100:.1f}%** | RPO **{w_rpo*100:.1f}%** | "
    f"Billings **{w_bill*100:.1f}%** | Cust **{w_cust*100:.1f}%**"
)
st.sidebar.markdown("---")

# Refresh data button
if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    with st.sidebar.status("Refreshing data & updating backtest models..."):
        updated = run_data_collectors()
        st.cache_data.clear()
    st.sidebar.success(f"Updated {len(updated)} components!")
    st.rerun()


# --- ONLINE CONSENSUS RETRIEVAL ---
@st.cache_data(ttl=3600)
def fetch_online_consensus_revenue(ticker: str = "DDOG") -> dict:
    """
    Fetches consensus revenue estimates for the target quarter from TradingView.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://www.tradingview.com/symbols/NASDAQ-{ticker}/forecast-price-target/"
    }

    try:
        url = "https://scanner.tradingview.com/america/scan"
        symbol = f"NASDAQ:{ticker}" if ":" not in ticker else ticker
        payload = {
            "symbols": {"tickers": [symbol]},
            "columns": [
                "revenue_forecast_next_fq",
                "revenue_forecast_fq",
                "revenue_fq",
                "revenue_forecast_next_fy",
                "price_target_average",
                "earnings_per_share_forecast_next_fq"
            ]
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("data", [])
            if rows:
                d = rows[0].get("d", [])
                rev_next_fq = d[0]
                rev_fq = d[1]
                val = rev_next_fq or rev_fq
                if val:
                    val_m = float(val) / 1e6
                    return {
                        "source": "TradingView",
                        "consensus_m": val_m,
                        "revenue_forecast_next_fq_m": float(rev_next_fq) / 1e6 if rev_next_fq else None,
                        "revenue_forecast_fq_m": float(rev_fq) / 1e6 if rev_fq else None,
                        "price_target_avg": d[4],
                        "eps_forecast_next_fq": d[5],
                        "is_live": True,
                        "error": None
                    }
    except Exception as e:
        log.warning(f"Direct TradingView scanner scrape attempt failed: {e}")

    try:
        page_url = f"https://www.tradingview.com/symbols/NASDAQ-{ticker}/forecast-price-target/"
        resp = requests.get(page_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            import re
            rev_matches = re.findall(r'revenue[^\d]*(\d+\.?\d*)\s*(B|M)USD', resp.text, re.IGNORECASE)
            for num_str, unit in rev_matches:
                num = float(num_str)
                val_m = num * 1000.0 if unit.upper() == "B" else num
                if val_m > 500:
                    return {
                        "source": "TradingView Direct Webpage Scrape",
                        "consensus_m": val_m,
                        "is_live": True,
                        "error": None
                    }
    except Exception as e:
        log.warning(f"TradingView HTML page scrape failed: {e}")

    return {
        "source": "Fallback Estimate",
        "consensus_m": 1240.0,
        "is_live": False,
        "error": None
    }


# --- EMPIRICAL LINEAR REGRESSION NOWCASTING & 4-Q FORECAST ENGINE ---
metric_col_map = {
    'Revenue QoQ Growth Rate (%)': {'col': 'revenue_qoq (%)', 'type': 'qoq', 'unit': '%', 'base': 'rev'},
    'RPO YoY Growth Rate (%)': {'col': 'rpo_yoy (%)', 'type': 'yoy', 'unit': '%', 'base': 'rpo'},
    'Billings YoY Growth Rate (%)': {'col': 'billings_yoy (%)', 'type': 'yoy', 'unit': '%', 'base': 'billings'},
    'Large Customer QoQ Growth Rate (%)': {'col': 'large_customer_qoq (%)', 'type': 'qoq', 'unit': '%', 'base': 'cust'},
}

def get_daily_rolling_series(source_name: str, window_days: int) -> pd.Series:
    if 'PYPL' in source_name:
        return s_pypl.resample('D').mean().ffill().rolling(f'{window_days}D', min_periods=1).mean()
    elif 'Tech Headlines' in source_name:
        return s_tech.resample('D').mean().ffill().rolling(f'{window_days}D', min_periods=1).mean()
    elif 'Product Releases' in source_name or 'Announcement' in source_name:
        return s_ann.resample('D').size().rolling(f'{window_days}D', min_periods=1).sum()
    else:
        raise ValueError(f"Unknown source: {source_name}")

# Filter backtested signal pairs
if corr_df is not None and not corr_df.empty:
    df_pairs = corr_df[
        corr_df['to_backtest'].astype(str).str.strip().str.upper().isin(['TRUE', '1'])
    ].copy().reset_index(drop=True)
else:
    df_pairs = pd.DataFrame([
        {'source': 'Product Releases Frequency', 'metric': 'Revenue QoQ Growth Rate (%)', 'optimal window': 90, 'optimal lag': 0},
        {'source': 'Tech Headlines thematic sentiment score', 'metric': 'RPO YoY Growth Rate (%)', 'optimal window': 30, 'optimal lag': 7},
        {'source': 'PYPL Stock Returns', 'metric': 'Billings YoY Growth Rate (%)', 'optimal window': 60, 'optimal lag': 14},
        {'source': 'PYPL Returns / Market Proxy', 'metric': 'Large Customer QoQ Growth Rate (%)', 'optimal window': 90, 'optimal lag': 14},
    ])

models_dict = {}
daily_results = {}

for idx, row in df_pairs.iterrows():
    source_name = str(row['source']).strip()
    metric_name = str(row['metric']).strip()
    window_days = int(row['optimal window'])
    lag_days = int(row['optimal lag'])
    pair_label = f"{source_name} -> {metric_name}"
    meta = metric_col_map.get(metric_name, {'col': 'revenue_qoq (%)', 'type': 'qoq', 'unit': '%', 'base': 'rev'})
    
    s_target = qtr_df.set_index('timestamp_utc')[meta['col']].dropna()
    s_rolling_full = get_daily_rolling_series(source_name, window_days)
    
    t_pred = pd.to_datetime(s_rolling_full.index, utc=True).tz_localize(None).astype('datetime64[s]').astype(np.int64).astype(np.float64) + 86400 * lag_days
    v_pred = s_rolling_full.values
    t_targ = pd.to_datetime(s_target.index, utc=True).tz_localize(None).astype('datetime64[s]').astype(np.int64).astype(np.float64)
    v_targ = s_target.values
    
    if len(t_pred) > 0 and len(t_targ) > 0:
        valid_mask = (t_targ >= t_pred[0]) & (t_targ <= t_pred[-1])
        if np.any(valid_mask):
            matched_idx = np.searchsorted(t_pred, t_targ[valid_mask], side='right') - 1
            x_train = v_pred[matched_idx].reshape(-1, 1)
            y_train = v_targ[valid_mask]
        else:
            matched_idx = np.clip(np.searchsorted(t_pred, t_targ, side='right') - 1, 0, len(v_pred) - 1)
            x_train = v_pred[matched_idx].reshape(-1, 1)
            y_train = v_targ
    else:
        x_train = np.array([]).reshape(0, 1)
        y_train = np.array([])
    
    if len(y_train) >= 2 and len(x_train) >= 2:
        model = LinearRegression().fit(x_train, y_train)
        r2 = r2_score(y_train, model.predict(x_train))
        rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
    else:
        model = LinearRegression()
        model.coef_ = np.array([0.0])
        model.intercept_ = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
        r2 = 0.0
        rmse = 1.0
    
    daily_x_vals = [s_rolling_full.loc[:d].iloc[-1] for d in target_dates]
    daily_x_arr = np.array(daily_x_vals).reshape(-1, 1)
    daily_growth_pred = model.predict(daily_x_arr)
    
    days_passed_in_q = np.arange(1, total_days_in_quarter + 1)
    days_passed_in_yr = np.array([d.dayofyear for d in target_dates])

    if meta['base'] == 'rev' and meta['type'] == 'qoq':
        implied_rev_curve = hist_rev * (1.0 + (days_passed_in_q / float(total_days_in_quarter)) * daily_growth_pred / 100.0)
    elif meta['base'] == 'rpo' and meta['type'] == 'yoy':
        implied_rev_curve = hist_rev * (1.0 + (days_passed_in_q / 365.0) * daily_growth_pred / 100.0)
    elif meta['base'] == 'billings' and meta['type'] == 'yoy':
        implied_rev_curve = hist_rev * (1.0 + (days_passed_in_q / 365.0) * daily_growth_pred / 100.0)
    elif meta['base'] == 'cust' and meta['type'] == 'qoq':
        implied_rev_curve = hist_rev * (1.0 + (days_passed_in_q / float(total_days_in_quarter)) * daily_growth_pred / 100.0)
    else:
        implied_rev_curve = hist_rev * (1.0 + (days_passed_in_q / float(total_days_in_quarter)) * daily_growth_pred / 100.0)
        
    models_dict[pair_label] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'rmse': rmse,
        'n_train': len(y_train)
    }
    
    daily_results[metric_name] = {
        'pair_label': pair_label,
        'source': source_name,
        'window': window_days,
        'lag': lag_days,
        'daily_x': daily_x_arr.flatten(),
        'daily_growth_pred': daily_growth_pred,
        'implied_rev_curve': implied_rev_curve,
        'rmse': rmse
    }

# Combine with user-selected weights
combined_rev_curve = (
    w_rev * daily_results['Revenue QoQ Growth Rate (%)']['implied_rev_curve'] +
    w_rpo * daily_results['RPO YoY Growth Rate (%)']['implied_rev_curve'] +
    w_bill * daily_results['Billings YoY Growth Rate (%)']['implied_rev_curve'] +
    w_cust * daily_results['Large Customer QoQ Growth Rate (%)']['implied_rev_curve']
)

# Uncertainty Band
rmse_rev_d = hist_rev * (daily_results['Revenue QoQ Growth Rate (%)']['rmse'] / 100.0)
rmse_rpo_d = hist_rev * ((total_days_in_quarter / 365.0) * daily_results['RPO YoY Growth Rate (%)']['rmse'] / 100.0)
rmse_bill_d = hist_rev * ((total_days_in_quarter / 365.0) * daily_results['Billings YoY Growth Rate (%)']['rmse'] / 100.0)
rmse_cust_d = hist_rev * (daily_results['Large Customer QoQ Growth Rate (%)']['rmse'] / 100.0)

combined_rmse_dollars = (
    w_rev * rmse_rev_d +
    w_rpo * rmse_rpo_d +
    w_bill * rmse_bill_d +
    w_cust * rmse_cust_d
)

band_fan = np.linspace(0.0, combined_rmse_dollars, total_days_in_quarter)
upper_band = combined_rev_curve + band_fan
lower_band = combined_rev_curve - band_fan

prior_yr_curve = np.linspace(prior_year_prev_rev, prior_year_rev, total_days_in_quarter)
prior_qtr_curve = np.linspace(hist_rev_prev, hist_rev, total_days_in_quarter)

# Point values for the current day
eval_idx = day_of_quarter - 1
implied_revenue = float(combined_rev_curve[eval_idx])
implied_rev_qoq_effective = ((implied_revenue - hist_rev) / hist_rev) * 100.0

# Base consensus revenue for target quarter from online source
consensus_data = fetch_online_consensus_revenue("DDOG")
consensus_rev = float(consensus_data.get("consensus_m", 1240.0))
consensus_source = consensus_data.get("source", "Analyst Consensus")
consensus_curve = np.linspace(hist_rev, consensus_rev, total_days_in_quarter)

# Deviation from consensus
deviation_dollars = implied_revenue - consensus_rev
deviation_pct = (deviation_dollars / consensus_rev) * 100.0

# Directional call on the quarter
if deviation_pct > 0.50:
    directional_call = "TRACKING AHEAD"
    badge_class = "badge-beat"
elif deviation_pct < -0.50:
    directional_call = "TRACKING BEHIND"
    badge_class = "badge-miss"
else:
    directional_call = "IN-LINE"
    badge_class = "badge-inline"

# Compute weighted average Risk-to-Reward (Signal Conviction) and weighted directional hit rate using sidebar weights
weight_map = {
    'Revenue QoQ Growth Rate (%)': w_rev,
    'RPO YoY Growth Rate (%)': w_rpo,
    'Billings YoY Growth Rate (%)': w_bill,
    'Large Customer QoQ Growth Rate (%)': w_cust,
}

fallback_bt_metrics = {
    'Revenue QoQ Growth Rate (%)': {'hit_rate': 57.89, 'avg_rr': 2.37},
    'RPO YoY Growth Rate (%)': {'hit_rate': 57.14, 'avg_rr': 1.89},
    'Billings YoY Growth Rate (%)': {'hit_rate': 69.23, 'avg_rr': 4.98},
    'Large Customer QoQ Growth Rate (%)': {'hit_rate': 41.18, 'avg_rr': 3.29},
}

weighted_avg_rr = 0.0
weighted_hit_rate = 0.0

for m_name, w in weight_map.items():
    hit_val = fallback_bt_metrics[m_name]['hit_rate']
    rr_val = fallback_bt_metrics[m_name]['avg_rr']
    if bt_df is not None and not bt_df.empty:
        match_row = bt_df[bt_df['metric'] == m_name]
        if not match_row.empty:
            hit_val = float(match_row.iloc[0]['directional_hit_rate (%)'])
            rr_val = float(match_row.iloc[0]['avg_risk_to_reward_ratio'])
    weighted_avg_rr += w * rr_val
    weighted_hit_rate += w * hit_val


# --- 4-QUARTER AHEAD FORECAST CALCULATIONS ---
forecast_results_4q = {q: {} for q in future_4_quarters}

for metric_name, meta in metric_col_map.items():
    res = daily_results[metric_name]
    pair_label = res['pair_label']
    m_info = models_dict[pair_label]
    model = m_info['model']
    source_name = res['source']
    window_days = res['window']
    
    s_rolling = get_daily_rolling_series(source_name, window_days)
    
    for q_label, q_date in zip(future_4_quarters, future_4_dates):
        sub = s_rolling.loc[:q_date]
        val = sub.iloc[-1] if len(sub) > 0 else s_rolling.iloc[-1]
        pred_g = float(model.predict([[val]])[0])
        
        forecast_results_4q[q_label][metric_name] = {
            'growth_pred': pred_g,
            'predictor_val': val,
            'source': source_name
        }

# Sequential and YoY level propagation across 4 quarters
rev_history_4q = list(qtr_df['revenue (USD)'].values / 1e6)
rpo_history_4q = list(qtr_df['rpo (USD)'].values / 1e6)
bill_history_4q = list(qtr_df['billings (USD)'].values / 1e6)
cust_history_4q = list(qtr_df['large_customer_count'].values)
labels_history_4q = list(qtr_df['quarter_label'].values)
n_hist = len(qtr_df)

forecast_4q_rows = []

for q_label in future_4_quarters:
    g_rev_qoq = forecast_results_4q[q_label]['Revenue QoQ Growth Rate (%)']['growth_pred']
    g_rpo_yoy = forecast_results_4q[q_label]['RPO YoY Growth Rate (%)']['growth_pred']
    g_bill_yoy = forecast_results_4q[q_label]['Billings YoY Growth Rate (%)']['growth_pred']
    g_cust_qoq = forecast_results_4q[q_label]['Large Customer QoQ Growth Rate (%)']['growth_pred']
    
    prev_rev = rev_history_4q[-1]
    prev_cust = cust_history_4q[-1]
    
    prior_y_rpo = rpo_history_4q[-4]
    prior_y_bill = bill_history_4q[-4]
    prior_y_rev = rev_history_4q[-4]
    
    f_rev = prev_rev * (1.0 + g_rev_qoq / 100.0)
    f_rpo = prior_y_rpo * (1.0 + g_rpo_yoy / 100.0)
    f_bill = prior_y_bill * (1.0 + g_bill_yoy / 100.0)
    f_cust = int(round(prev_cust * (1.0 + g_cust_qoq / 100.0)))
    
    impl_rev_from_rev = prev_rev * (1.0 + g_rev_qoq / 100.0)
    impl_rev_from_rpo = prev_rev * (1.0 + 0.25 * g_rpo_yoy / 100.0)
    impl_rev_from_bill = prev_rev * (1.0 + 0.25 * g_bill_yoy / 100.0)
    impl_rev_from_cust = prev_rev * (1.0 + g_cust_qoq / 100.0)
    
    f_combined_rev = (
        w_rev * impl_rev_from_rev +
        w_rpo * impl_rev_from_rpo +
        w_bill * impl_rev_from_bill +
        w_cust * impl_rev_from_cust
    )
    
    f_rev_yoy = ((f_combined_rev - prior_y_rev) / prior_y_rev) * 100.0
    
    rev_history_4q.append(f_combined_rev)
    rpo_history_4q.append(f_rpo)
    bill_history_4q.append(f_bill)
    cust_history_4q.append(f_cust)
    labels_history_4q.append(q_label)
    
    forecast_4q_rows.append({
        'Quarter': q_label,
        'Combined Forecast Rev ($M)': f"${f_combined_rev:,.2f}M",
        'Forecast Rev YoY Growth': f"{f_rev_yoy:+.1f}%",
        'Forecast RPO ($M)': f"${f_rpo:,.1f}M",
        'Forecast Billings ($M)': f"${f_bill:,.1f}M",
        'Forecast Large Customers': f"{f_cust:,}",
    })

df_4q_summary = pd.DataFrame(forecast_4q_rows)


# --- STATUS AS OF DATE CALCULATION ---
status_date = target_q_start_date + timedelta(days=day_of_quarter - 1)
date_formatted = f"{status_date.day} {status_date.strftime('%b %Y')}"
status_as_of_str = f"Status as of {target_quarter_label} Day {day_of_quarter}/{total_days_in_quarter} ({date_formatted})"

# --- HEADER SECTION ---
st.markdown('<div class="main-header">Datadog (NASDAQ: DDOG) Revenue Nowcasting Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{status_as_of_str}</div>', unsafe_allow_html=True)


# --- KPI CARDS ROW ---
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title" title="Combined Revenue Nowcast">Combined Rev. Nowcast</div>
        <div class="metric-value" style="color: #2563EB;">${implied_revenue:.1f}M</div>
        <div class="metric-footer" style="color: #2563EB;">{implied_rev_qoq_effective:+.1f}% QoQ Growth</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    deviation_color = "#15803D" if deviation_dollars >= 0 else "#B91C1C"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title" title="Directional Call">Directional Call</div>
        <div style="margin: 2px 0 6px 0;">
            <span class="{badge_class}">{directional_call}</span>
        </div>
        <div class="metric-footer" style="color: {deviation_color};">{deviation_pct:+.2f}% vs consensus ({consensus_source})</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title" title="Weighted Average Risk-to-Reward Ratio">Risk-to-Reward ratio</div>
        <div class="metric-value" style="color: #0284C7;">{weighted_avg_rr:.2f}</div>
        <div class="metric-footer" style="color: #64748B;">Hit Rate: {weighted_hit_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Intra-Quarter Tracking",
    "📈 4-Quarter Ahead Forecast",
    "🔄 Signal Data Latency",
    "🕰️ Historical Backtesting"
])

# -------------------------------------------------------------
# TAB 1: INTRA-QUARTER TRACKING
# -------------------------------------------------------------
with tab1:
    st.subheader(f"Intra-Quarter Tracking of Combined Revenue Estimate")
    st.caption("Tracking ahead (behind) means our estimated trajectory has exceeded (fallen below) the consensus trajectory.")

    days = np.arange(1, total_days_in_quarter + 1)
    curr_days = np.arange(1, day_of_quarter + 1)
    curr_nowcast_curve = combined_rev_curve[:day_of_quarter]

    fig_combined = go.Figure()

    # Shaded error band
    fig_combined.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([upper_band, lower_band[::-1]]),
        fill='toself',
        fillcolor='rgba(203, 213, 225, 0.35)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'±1x Combined RMSE (±${combined_rmse_dollars:.1f}M)'
    ))

    # Prior year comparison (YoY growth)
    fig_combined.add_trace(go.Scatter(
        x=days,
        y=prior_yr_curve,
        mode='lines',
        line=dict(color='#94a3b8', dash='dot', width=1.8),
        name=f'Prior year (${prior_year_rev:.1f}M)'
    ))

    # Prior quarter comparison (QoQ growth)
    fig_combined.add_trace(go.Scatter(
        x=days,
        y=prior_qtr_curve,
        mode='lines',
        line=dict(color='#64748b', dash='dash', width=1.8),
        name=f'Prior quarter (${hist_rev:.1f}M)'
    ))

    # Baseline from analysts' consensus on TradingView
    fig_combined.add_trace(go.Scatter(
        x=days,
        y=consensus_curve,
        mode='lines',
        line=dict(color='#D97706', width=2.2, dash='dash'),
        name=f'Consensus (${consensus_rev:.1f}M)'
    ))

    # Nowcasted full path
    fig_combined.add_trace(go.Scatter(
        x=days,
        y=combined_rev_curve,
        mode='lines',
        line=dict(color='#2563eb', dash='dot', width=2.2),
        name=f'Nowcasted full path (${combined_rev_curve[-1]:.1f}M)'
    ))

    # In-progress pace curve up to day N
    pace_line_color = '#15803D' if deviation_dollars >= 0 else '#B91C1C'
    fig_combined.add_trace(go.Scatter(
        x=curr_days,
        y=curr_nowcast_curve,
        mode='lines',
        line=dict(color=pace_line_color, width=3.4),
        name=f'Nowcast pace [Day {day_of_quarter}]'
    ))

    # Current day marker
    fig_combined.add_trace(go.Scatter(
        x=[day_of_quarter],
        y=[curr_nowcast_curve[-1]],
        mode='markers+text',
        marker=dict(color=pace_line_color, size=11, symbol='circle'),
        text=[f"${curr_nowcast_curve[-1]:.1f}M (Day {day_of_quarter})"],
        textposition="top center",
        name='Current nowcast point'
    ))

    fig_combined.update_layout(
        title=dict(
            text=f"<b>Datadog Intra-Quarter Combined Revenue Estimate</b>",
            font=dict(size=14),
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top"
        ),
        xaxis_title="Day of fiscal quarter",
        yaxis_title="Combined revenue estimate (USD Millions)",
        yaxis=dict(tickprefix="$", ticksuffix="M"),
        xaxis=dict(range=[1, total_days_in_quarter]),
        hovermode="x unified",
        template="plotly_white",
        height=490,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10)
        )
    )

    st.plotly_chart(fig_combined, use_container_width=True)

    # Table breakdown of signal contributions
    st.markdown('<h4 style="margin-top: 0px; margin-bottom: 8px; font-size: 1.25rem; font-weight: 600;">🔍 Breakdown of signal contributions</h4>', unsafe_allow_html=True)

    res_rev = daily_results['Revenue QoQ Growth Rate (%)']
    res_rpo = daily_results['RPO YoY Growth Rate (%)']
    res_bill = daily_results['Billings YoY Growth Rate (%)']
    res_cust = daily_results['Large Customer QoQ Growth Rate (%)']

    breakdown_data = [
        {
            "Metric component": "Revenue QoQ Growth Rate (%)",
            "Signal": res_rev['source'],
            "Window / Lag": f"{res_rev['window']}D / {res_rev['lag']}D",
            "Predictor Value": f"{res_rev['daily_x'][eval_idx]:.2f}",
            "Predicted Momentum": f"{res_rev['daily_growth_pred'][eval_idx]:+.2f}%",
            "Implied Revenue ($M)": f"${res_rev['implied_rev_curve'][eval_idx]:.1f}M",
            "Weight": f"{w_rev*100:.1f}%",
            "Revenue Contribution ($M)": f"${(w_rev * res_rev['implied_rev_curve'][eval_idx]):.1f}M"
        },
        {
            "Metric component": "RPO YoY Growth Rate (%)",
            "Signal": res_rpo['source'],
            "Window / Lag": f"{res_rpo['window']}D / {res_rpo['lag']}D",
            "Predictor Value": f"{res_rpo['daily_x'][eval_idx]:.2f}",
            "Predicted Momentum": f"{res_rpo['daily_growth_pred'][eval_idx]:+.2f}%",
            "Implied Revenue ($M)": f"${res_rpo['implied_rev_curve'][eval_idx]:.1f}M",
            "Weight": f"{w_rpo*100:.1f}%",
            "Revenue Contribution ($M)": f"${(w_rpo * res_rpo['implied_rev_curve'][eval_idx]):.1f}M"
        },
        {
            "Metric component": "Billings YoY Growth Rate (%)",
            "Signal": res_bill['source'],
            "Window / Lag": f"{res_bill['window']}D / {res_bill['lag']}D",
            "Predictor Value": f"{res_bill['daily_x'][eval_idx]:.2f}",
            "Predicted Momentum": f"{res_bill['daily_growth_pred'][eval_idx]:+.2f}%",
            "Implied Revenue ($M)": f"${res_bill['implied_rev_curve'][eval_idx]:.1f}M",
            "Weight": f"{w_bill*100:.1f}%",
            "Revenue Contribution ($M)": f"${(w_bill * res_bill['implied_rev_curve'][eval_idx]):.1f}M"
        },
        {
            "Metric component": "Large Customer Count QoQ Growth (%)",
            "Signal": res_cust['source'],
            "Window / Lag": f"{res_cust['window']}D / {res_cust['lag']}D",
            "Predictor Value": f"{res_cust['daily_x'][eval_idx]:.2f}",
            "Predicted Momentum": f"{res_cust['daily_growth_pred'][eval_idx]:+.2f}%",
            "Implied Revenue ($M)": f"${res_cust['implied_rev_curve'][eval_idx]:.1f}M",
            "Weight": f"{w_cust*100:.1f}%",
            "Revenue Contribution ($M)": f"${(w_cust * res_cust['implied_rev_curve'][eval_idx]):.1f}M"
        },
    ]

    df_breakdown = pd.DataFrame(breakdown_data)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    # Mathematical Formulas
    st.markdown("#### 📐 Mathematical formulation of combined revenue estimate")
    st.markdown(r"""
    The intra-quarter implied revenue of each individual metric component at day $$t \in \{1, \dots, N_{\text{days}}\}$$ of the quarter $q$ is calculated as:
    * **Revenue QoQ Growth Rate (%)**:
      $$\widehat{\text{Rev}}_{\text{rev}}(t) = \text{Rev}_{q-1} \times \left(1 + \frac{t}{N_{\text{days}}} \cdot \frac{\hat{g}_{\text{rev, QoQ}}(t)}{100}\right)$$
    * **RPO YoY Growth Rate (%)**:
      $$\widehat{\text{Rev}}_{\text{rpo}}(t) = \text{Rev}_{q-1} \times \left(1 + \frac{t}{365} \cdot \frac{\hat{g}_{\text{rpo, YoY}}(t)}{100}\right)$$
    * **Billings YoY Growth Rate (%)**:
      $$\widehat{\text{Rev}}_{\text{bill}}(t) = \text{Rev}_{q-1} \times \left(1 + \frac{t}{365} \cdot \frac{\hat{g}_{\text{bill, YoY}}(t)}{100}\right)$$
    * **Large Customer Count QoQ Growth (%)**:
      $$\widehat{\text{Rev}}_{\text{cust}}(t) = \text{Rev}_{q-1} \times \left(1 + \frac{t}{N_{\text{days}}} \cdot \frac{\hat{g}_{\text{cust, QoQ}}(t)}{100}\right)$$
    where $\text{Rev}_{q-1}$ is the prior quarter's reported revenue, $N_{\text{days}}$ is the total number of days in the ongoing quarter, and $\hat{g} refers to predicted growth rate from a linear regression model.
    
    The combined revenue estimate is then given by:

    $$\widehat{\text{Rev}}_{\text{combined}}(t) = w_{\text{rev}} \cdot \widehat{\text{Rev}}_{\text{rev}}(t) + w_{\text{rpo}} \cdot \widehat{\text{Rev}}_{\text{rpo}}(t) + w_{\text{bill}} \cdot \widehat{\text{Rev}}_{\text{bill}}(t) + w_{\text{cust}} \cdot \widehat{\text{Rev}}_{\text{cust}}(t)$$

    where $w_i$ are the component weights defined in the sidebar.
    """)


# -------------------------------------------------------------
# TAB 2: 4-QUARTER AHEAD FORECAST
# -------------------------------------------------------------
with tab2:
    st.subheader(f"Combined Revenue 4-Quarter Ahead Forecast Trajectory ({future_4_quarters[0]} – {future_4_quarters[-1]})")
    st.caption(f"Continuous quarterly revenue trajectory synthesized from 4 leading indicator empirical linear regression models (Weights: {w_rev*100:.0f}% Rev QoQ | {w_rpo*100:.0f}% RPO YoY | {w_bill*100:.0f}% Billings YoY | {w_cust*100:.0f}% Cust QoQ)")

    # Main 4-Quarter Forecast Combined Trajectory Chart
    fig_4q = go.Figure()

    # Historical Reported Revenue
    fig_4q.add_trace(go.Scatter(
        x=labels_history_4q[:n_hist],
        y=rev_history_4q[:n_hist],
        mode='lines+markers',
        marker=dict(size=6, color='#2563EB'),
        line=dict(color='#2563EB', width=2.6),
        name='Historical Reported Revenue (SEC)'
    ))

    # 4-Quarter Forecast Path
    fig_4q.add_trace(go.Scatter(
        x=labels_history_4q[n_hist-1:],
        y=rev_history_4q[n_hist-1:],
        mode='lines+markers',
        marker=dict(size=8, color='#16A34A', symbol='diamond'),
        line=dict(color='#16A34A', width=3.0, dash='solid'),
        name=f'Combined 4-Quarter Revenue Forecast ({future_4_quarters[0]} - {future_4_quarters[-1]})'
    ))

    fig_4q.add_shape(
        type="line",
        x0=labels_history_4q[n_hist-1],
        x1=labels_history_4q[n_hist-1],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#64748B", width=1.5, dash="dot")
    )
    fig_4q.add_annotation(
        x=labels_history_4q[n_hist-1],
        y=1.0,
        yref="paper",
        text="Forecast Horizon →",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10.5, color="#475569")
    )

    fig_4q.update_layout(
        title=dict(
            text="<b>Datadog (NASDAQ: DDOG) Combined Revenue Forecast: Historical Reported vs Next 4 Quarters</b>",
            font=dict(size=14.5)
        ),
        xaxis_title="Fiscal Quarter",
        yaxis_title="Combined Revenue (USD Millions)",
        yaxis=dict(tickprefix="$", ticksuffix="M"),
        hovermode="x unified",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10.5))
    )

    st.plotly_chart(fig_4q, use_container_width=True)

    # 4-Quarter Summary Table
    st.markdown("#### 📋 4-Quarter Ahead Metric Predictions Breakdown")
    st.dataframe(df_4q_summary, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔍 4-Quarter Trajectories for Individual Metric Pairs")

    # 2x2 Layout for the 4 Individual Pairs
    col_p1, col_p2 = st.columns(2)

    def make_pair_plotly(history_vals, title, metric_label, pred_growth_str, color_hex, is_dollar=True):
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=labels_history_4q[:n_hist],
            y=history_vals[:n_hist],
            mode='lines+markers',
            marker=dict(size=5, color=color_hex),
            line=dict(color=color_hex, width=2.2),
            name='Historical Reported'
        ))
        fig_p.add_trace(go.Scatter(
            x=labels_history_4q[n_hist-1:],
            y=history_vals[n_hist-1:],
            mode='lines+markers',
            marker=dict(size=7, color='#D97706', symbol='square'),
            line=dict(color='#D97706', width=2.4, dash='dash'),
            name=f'4-Quarter Forecast ({pred_growth_str})'
        ))

        fig_p.add_shape(
            type="line",
            x0=labels_history_4q[n_hist-1],
            x1=labels_history_4q[n_hist-1],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#94A3B8", width=1.2, dash="dot")
        )

        fig_p.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=12)
            ),
            xaxis_title="Fiscal Quarter",
            yaxis_title=metric_label,
            yaxis=dict(tickprefix="$" if is_dollar else "", ticksuffix="M" if is_dollar else ""),
            hovermode="x unified",
            template="plotly_white",
            height=320,
            margin=dict(l=15, r=15, t=45, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
        )
        return fig_p

    # Top 2 plots
    with col_p1:
        g1 = forecast_results_4q[future_4_quarters[0]]['Revenue QoQ Growth Rate (%)']['growth_pred']
        fig1 = make_pair_plotly(
            rev_history_4q,
            "Product Releases → Revenue ($M)",
            "Revenue ($M)",
            f"Pred QoQ: {g1:+.2f}%",
            "#7C3AED",
            is_dollar=True
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_p2:
        g2 = forecast_results_4q[future_4_quarters[0]]['RPO YoY Growth Rate (%)']['growth_pred']
        fig2 = make_pair_plotly(
            rpo_history_4q,
            "Tech Headlines → RPO ($M)",
            "RPO ($M)",
            f"Pred YoY: {g2:+.2f}%",
            "#EA580C",
            is_dollar=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Bottom 2 plots
    col_p3, col_p4 = st.columns(2)

    with col_p3:
        g3 = forecast_results_4q[future_4_quarters[0]]['Billings YoY Growth Rate (%)']['growth_pred']
        fig3 = make_pair_plotly(
            bill_history_4q,
            "PYPL Returns → Calculated Billings ($M)",
            "Billings ($M)",
            f"Pred YoY: {g3:+.2f}%",
            "#16A34A",
            is_dollar=True
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_p4:
        g4 = forecast_results_4q[future_4_quarters[0]]['Large Customer QoQ Growth Rate (%)']['growth_pred']
        fig4 = make_pair_plotly(
            cust_history_4q,
            "PYPL Returns → Large Customers (ARR ≥ $100k)",
            "Large Customer Count",
            f"Pred QoQ: {g4:+.2f}%",
            "#0284C7",
            is_dollar=False
        )
        st.plotly_chart(fig4, use_container_width=True)


# -------------------------------------------------------------
# TAB 3: SIGNAL DATA LATENCY
# -------------------------------------------------------------
with tab3:
    st.subheader("Signal Data Latency")

    dataset_latency_data = get_dataset_latency_metadata()
    latency_df = pd.DataFrame(dataset_latency_data)
    st.dataframe(latency_df, use_container_width=True, hide_index=True)

    # Ingestion architecture
    st.markdown("#### Real-Time Alternative Data Ingestion Architecture")
    st.markdown("""
    ```
    [Market closing prices] ───────────▶ [PYPL stock returns] ──────────────────────────────────▶ [Rolling average] ───┐ 
    [Hourly tech headlines scraper] ──▶ [SetFit NLP model] ───▶ [Thematic sentiment score] ───▶ [Rolling average] ───┼──▶ [DDOG Nowcast Engine]
    [Daily Datadog newsroom scraper] ──▶ [Keyword classification of product releases] ────────▶ [Rolling pace]    ───┘
    ```
    """)


# -------------------------------------------------------------
# TAB 4: HISTORICAL BACKTESTING
# -------------------------------------------------------------
with tab4:
    st.subheader("Historical Walk-Forward Backtesting & Model Performance")

    if bt_df is not None:
        st.markdown("#### Summary of Out-of-Sample Backtesting Metrics")
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        
        # Display latest quarter backtested
        latest_bt_qtr = qtr_df['quarter_label'].iloc[-1] if qtr_df is not None and not qtr_df.empty else "2026 Q2"
        st.markdown(f"**Latest Quarter Backtested:** `{latest_bt_qtr}` *(Backtests re-evaluated automatically on quarterly data updates)*")

    # --- COMPARISON OF PREDICTED VS ACTUAL TRAJECTORIES ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Comparison of Predicted vs Actual Trajectories")
    st.caption("Out-of-sample walk-forward backtesting predictions vs actual reported quarterly growth rates for the 4 alternative data signal pairs. Green checkmarks (✓) denote correct directional growth calls, while red crosses (✗) denote misses.")

    if bt_detailed_df is not None and not bt_detailed_df.empty:
        pair_colors = {
            'Revenue QoQ Growth Rate (%)': '#3182bd',
            'RPO YoY Growth Rate (%)': '#fd8d3c',
            'Billings YoY Growth Rate (%)': '#74c476',
            'Large Customer QoQ Growth Rate (%)': '#756bb1',
        }

        unique_pairs = list(bt_detailed_df['pair_name'].unique())

        # Render 2x2 grid of comparison plots
        for i in range(0, len(unique_pairs), 2):
            col_bt1, col_bt2 = st.columns(2)
            pair_cols = [col_bt1, col_bt2]

            for j in range(2):
                if i + j < len(unique_pairs):
                    pair_name = unique_pairs[i + j]
                    with pair_cols[j]:
                        df_pair = bt_detailed_df[bt_detailed_df['pair_name'] == pair_name].sort_values('date')
                        metric_name = df_pair['metric'].iloc[0]
                        line_color = pair_colors.get(metric_name, '#2563EB')

                        # Fetch summary metrics
                        summary_match = pd.DataFrame()
                        if bt_df is not None and not bt_df.empty:
                            summary_match = bt_df[bt_df.apply(lambda r: f"{r['source']} -> {r['metric']}" == pair_name, axis=1)]

                        if not summary_match.empty:
                            s_row = summary_match.iloc[0]
                            title_text = f"<b>{pair_name}</b>"
                            subtitle_text = (
                                f"Window={s_row['optimal window']}d, Lag={s_row['optimal lag']}d | "
                                f"MAPE: {s_row['mape (%)']:.1f}% | RMSE: {s_row['rmse']:.2f} | "
                                f"Hit Rate: {s_row['directional_hit_rate (%)']:.1f}% | "
                                f"Avg R/R: {s_row['avg_risk_to_reward_ratio']:.2f}"
                            )
                        else:
                            title_text = f"<b>{pair_name}</b>"
                            subtitle_text = "Walk-Forward OOS Trajectory"

                        fig_pair = go.Figure()

                        # Actual Values Trace
                        fig_pair.add_trace(go.Scatter(
                            x=df_pair['date'],
                            y=df_pair['actual'],
                            mode='lines+markers',
                            name='Actual',
                            marker=dict(size=7, color=line_color, symbol='circle'),
                            line=dict(color=line_color, width=2.4),
                            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Actual:</b> %{y:.2f}%<extra></extra>"
                        ))

                        # Predicted Values Trace
                        fig_pair.add_trace(go.Scatter(
                            x=df_pair['date'],
                            y=df_pair['predicted'],
                            mode='lines+markers',
                            name='Predicted',
                            marker=dict(size=7, color='#d95f02', symbol='square'),
                            line=dict(color='#d95f02', width=2.0, dash='dash'),
                            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Predicted:</b> %{y:.2f}%<extra></extra>"
                        ))

                        # Annotate Hit / Miss Directional Calls
                        for _, row_pt in df_pair.iterrows():
                            d_pt = row_pt['date']
                            pred_val = row_pt['predicted']
                            is_hit = bool(row_pt['correct_direction'])
                            symbol = "✓" if is_hit else "✗"
                            symbol_color = "#16A34A" if is_hit else "#DC2626"

                            fig_pair.add_annotation(
                                x=d_pt,
                                y=pred_val,
                                text=f"<b>{symbol}</b>",
                                showarrow=False,
                                yshift=14,
                                font=dict(size=12, color=symbol_color)
                            )

                        fig_pair.update_layout(
                            title=dict(
                                text=f"{title_text}<br><span style='font-size: 11px; font-weight: normal; color: #64748B;'>{subtitle_text}</span>",
                                font=dict(size=13)
                            ),
                            xaxis_title="Quarterly Out-of-Sample Date",
                            yaxis_title=metric_name,
                            yaxis=dict(ticksuffix="%"),
                            hovermode="x unified",
                            template="plotly_white",
                            height=360,
                            margin=dict(l=20, r=20, t=60, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
                        )

                        st.plotly_chart(fig_pair, use_container_width=True)


# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 12px;'>"
    f"Datadog Quarterly Metrics Nowcasting Dashboard | {now_dt.strftime('%d %B %Y')}"
    "</div>",
    unsafe_allow_html=True
)
