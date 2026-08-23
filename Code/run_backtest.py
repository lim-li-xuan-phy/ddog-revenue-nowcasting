'''
run_backtest.py
===============
Executes out-of-sample (OOS) walk-forward backtesting for the alternative data signals
and updates Results/Summaries/backtesting_summary.csv automatically.

Signals tested:
1. Product Releases Frequency -> Revenue QoQ Growth Rate (%)
2. Tech Headlines Score -> RPO YoY Growth Rate (%)
3. PYPL Returns -> Billings YoY Growth Rate (%)
4. PYPL Returns -> Large Customer QoQ Growth Rate (%)
'''

import logging
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

CODE_DIR = Path(__file__).parent.resolve()
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
SUMMARIES_DIR = RESULTS_DIR / "Summaries"


metric_col_map = {
    'Revenue QoQ Growth Rate (%)': {'col': 'revenue_qoq (%)', 'color': '#3182bd', 'type': 'pct'},
    'RPO YoY Growth Rate (%)': {'col': 'rpo_yoy (%)', 'color': '#fd8d3c', 'type': 'pct'},
    'Billings YoY Growth Rate (%)': {'col': 'billings_yoy (%)', 'color': '#74c476', 'type': 'pct'},
    'Large Customer QoQ Growth Rate (%)': {'col': 'large_customer_qoq (%)', 'color': '#756bb1', 'type': 'pct'},
    'Revenue YoY Growth Rate (%)': {'col': 'revenue_yoy (%)', 'color': '#6baed6', 'type': 'pct'},
    'RPO QoQ Growth Rate (%)': {'col': 'rpo_qoq (%)', 'color': '#e6550d', 'type': 'pct'},
    'Billings QoQ Growth Rate (%)': {'col': 'billings_qoq (%)', 'color': '#31a354', 'type': 'pct'},
    'Large Customer YoY Growth Rate (%)': {'col': 'large_customer_yoy (%)', 'color': '#9e9ac8', 'type': 'pct'},
}


def extract_time_and_values(data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pd.Series):
        values = data.to_numpy(dtype=np.float64)
        if isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index.astype(np.int64) / 1e9
        else:
            timestamps = np.asarray(data.index, dtype=np.float64)
    else:
        raise ValueError("Expected pd.Series.")
    sort_idx = np.argsort(timestamps)
    return timestamps[sort_idx], values[sort_idx]


def align_timestamps(s1: pd.Series, s2: pd.Series, s1_lag_days: int = 0):
    if s1_lag_days is None:
        s1_lag_days = 0

    t1, raw_data1 = extract_time_and_values(s1)
    t2, raw_data2 = extract_time_and_values(s2)

    # Shift s1 timestamps forward by lag_days
    t1 = t1 + 86400 * s1_lag_days

    start_time = max(t1[0], t2[0])
    end_time = min(t1[-1], t2[-1])

    valid_mask2 = (t2 >= start_time) & (t2 <= end_time)
    valid_t2 = t2[valid_mask2]
    data2 = raw_data2[valid_mask2]
    dates2 = s2.index[valid_mask2]

    matched_idx = np.searchsorted(t1, valid_t2, side='right') - 1
    data1 = raw_data1[matched_idx]

    return data1, data2, dates2


def run_walk_forward_backtest(
    s_predictor: pd.Series,
    s_target: pd.Series,
    lag_days: int,
    initial_train_size: int = 10,
):
    x, y, dates = align_timestamps(s_predictor, s_target, lag_days)
    n_samples = len(y)

    if n_samples <= initial_train_size:
        raise ValueError(f"Not enough samples ({n_samples}) for initial train size {initial_train_size}")

    oos_preds = []
    oos_actuals = []
    oos_dates = []
    oos_prior_actuals = []

    for t in range(initial_train_size, n_samples):
        x_train = x[:t].reshape(-1, 1)
        y_train = y[:t]

        x_test = x[t].reshape(-1, 1)
        y_test = y[t]

        model = LinearRegression().fit(x_train, y_train)
        pred = model.predict(x_test)[0]

        oos_preds.append(pred)
        oos_actuals.append(y_test)
        oos_dates.append(dates[t])
        oos_prior_actuals.append(y[t - 1])

    oos_preds = np.array(oos_preds)
    oos_actuals = np.array(oos_actuals)
    oos_prior_actuals = np.array(oos_prior_actuals)

    eps = 1e-6
    safe_actuals = np.where(np.abs(oos_actuals) < eps, eps, oos_actuals)
    mape = float(np.mean(np.abs((oos_actuals - oos_preds) / safe_actuals)) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(oos_actuals, oos_preds)))

    actual_change = oos_actuals - oos_prior_actuals
    pred_change = oos_preds - oos_prior_actuals

    correct_dir = (np.sign(pred_change) == np.sign(actual_change))
    directional_hit_rate = float(np.mean(correct_dir) * 100.0)

    errors = np.abs(oos_actuals - oos_preds)
    actual_move_magnitudes = np.abs(actual_change)

    avg_loss_magnitude = float(np.mean(errors[~correct_dir])) if np.sum(~correct_dir) > 0 else 0.0
    avg_win_magnitude = float(np.mean(actual_move_magnitudes[correct_dir])) if np.sum(correct_dir) > 0 else 1e-6

    avg_risk_to_reward = float(avg_loss_magnitude / avg_win_magnitude) if avg_win_magnitude > 0 else 0.0

    df_oos_details = pd.DataFrame({
        'date': oos_dates,
        'actual': oos_actuals,
        'predicted': oos_preds,
        'prior_actual': oos_prior_actuals,
        'actual_change': actual_change,
        'pred_change': pred_change,
        'correct_direction': correct_dir,
        'error': errors
    })

    latest_date_tested = oos_dates[-1] if oos_dates else None

    metrics = {
        'mape': mape,
        'rmse': rmse,
        'directional_hit_rate': directional_hit_rate,
        'avg_risk_to_reward': avg_risk_to_reward,
        'n_oos_quarters': len(oos_actuals),
        'latest_date_tested': latest_date_tested,
    }

    return metrics, df_oos_details


def plot_backtest_trajectories(
    detailed_oos_records: dict,
    df_summary: pd.DataFrame,
    output_path: Path = RESULTS_DIR / 'walkforward_backtest.png'
) -> Path:
    """
    Plots the out-of-sample walk-forward backtest predicted vs actual trajectories
    for each signal pair with directional hit/miss annotations, matching Backtesting.ipynb.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n_pairs = len(detailed_oos_records)
    if n_pairs == 0:
        return output_path

    fig, axes = plt.subplots(n_pairs, 1, figsize=(13, 3.8 * n_pairs), sharex=False)
    if n_pairs == 1:
        axes = [axes]

    for i, (pair_name, (df_detail, metric_info)) in enumerate(detailed_oos_records.items()):
        ax = axes[i]
        color = metric_info.get('color', '#3182bd')
        dates = df_detail['date']

        # Plot actual values
        ax.plot(dates, df_detail['actual'], marker='o', color=color, linewidth=2.2, label='Actual')

        # Plot predicted values
        ax.plot(dates, df_detail['predicted'], marker='s', linestyle='--', color='#d95f02', linewidth=1.8, label='Predicted')

        # Annotate hit / miss directional calls
        for _, row in df_detail.iterrows():
            d = row['date']
            pred_val = row['predicted']
            is_hit = row['correct_direction']
            marker_symbol = '✓' if is_hit else '✗'
            marker_color = '#2ca02c' if is_hit else '#d62728'
            ax.annotate(
                marker_symbol,
                (d, pred_val),
                textcoords="offset points",
                xytext=(0, 9),
                ha='center',
                fontsize=10,
                fontweight='bold',
                color=marker_color
            )

        matching_rows = df_summary[df_summary.apply(lambda r: f"{r['source']} -> {r['metric']}" == pair_name, axis=1)]
        if not matching_rows.empty:
            summary_row = matching_rows.iloc[0]
            title_str = (
                f"{pair_name}\n"
                f"Window={summary_row['optimal window']}d, Lag={summary_row['optimal lag']}d | "
                f"MAPE: {summary_row['mape (%)']:.2f}% | RMSE: {summary_row['rmse']:.2f} | "
                f"Hit rate: {summary_row['directional_hit_rate (%)']:.1f}% | Avg risk-to-reward: {summary_row['avg_risk_to_reward_ratio']:.2f}"
            )
            ylabel_str = str(summary_row['metric'])
        else:
            title_str = pair_name
            ylabel_str = "Growth Rate (%)"

        ax.set_title(title_str, fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel_str, fontsize=9, fontweight='semibold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:.1f}%'))
        ax.legend(loc='best', fontsize=9)

    plt.suptitle("Walk-forward backtesting", fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved walk-forward backtest plot to {output_path}")
    return output_path


def run() -> Tuple[pd.DataFrame, str, dict]:
    """
    Executes walk-forward backtests on all primary signals, generates trajectory plot,
    and updates Results/Summaries/backtesting_summary.csv.
    Returns (df_summary, latest_quarter_backtested, detailed_oos_records).
    """
    log.info("Running automated walk-forward backtesting pipeline...")

    # 1. Load metrics
    metrics_csv = DATA_DIR / 'qtrly_metrics.csv'
    if not metrics_csv.exists():
        log.warning("qtrly_metrics.csv not found. Skipping backtest.")
        return pd.DataFrame(), "N/A", {}

    df_metrics = pd.read_csv(metrics_csv)
    df_metrics['timestamp_utc'] = pd.to_datetime(df_metrics['timestamp_utc'])
    df_metrics_idx = df_metrics.set_index('timestamp_utc').sort_index()

    latest_qtr_label = df_metrics['quarter_label'].iloc[-1] if 'quarter_label' in df_metrics.columns else "2026 Q2"

    # 2. Load alternative data series
    pypl_csv = DATA_DIR / 'pypl_returns.csv'
    tech_csv = DATA_DIR / 'tech_headlines.csv'
    ann_csv = DATA_DIR / 'announcements.csv'

    s_pypl = pd.read_csv(pypl_csv, parse_dates=['timestamp_utc']).set_index('timestamp_utc')['return'].dropna().sort_index()
    s_tech = pd.read_csv(tech_csv, parse_dates=['timestamp_utc']).set_index('timestamp_utc')['news_score'].dropna().sort_index()
    
    df_ann = pd.read_csv(ann_csv, parse_dates=['timestamp_utc'])
    # Normalize boolean product_release
    if 'product_release' in df_ann.columns:
        is_true_mask = df_ann['product_release'].astype(str).str.strip().str.upper().isin(['TRUE', '1'])
        s_ann = df_ann[is_true_mask].set_index('timestamp_utc')['product_release'].sort_index()
    else:
        s_ann = pd.Series(dtype=float)

    def get_rolling_pypl(window_days: int) -> pd.Series:
        return s_pypl.rolling(f'{window_days}D', min_periods=1).mean()

    def get_rolling_tech_headlines(window_days: int) -> pd.Series:
        return s_tech.rolling(f'{window_days}D', min_periods=1).mean()

    def get_rolling_announcements(window_days: int) -> pd.Series:
        return s_ann.resample('D').size().rolling(f'{window_days}D', min_periods=1).sum()

    # Backtest configurations
    backtest_pairs = [
        {
            'source': 'Product Releases Frequency',
            'metric': 'Revenue QoQ Growth Rate (%)',
            'optimal window': 71,
            'optimal lag': 88,
            'hy estimator': -0.6345,
            'predictor_func': lambda: get_rolling_announcements(71),
            'target_col': 'revenue_qoq (%)',
        },
        {
            'source': 'Tech Headlines Score',
            'metric': 'RPO YoY Growth Rate (%)',
            'optimal window': 9,
            'optimal lag': 71,
            'hy estimator': -0.2310,
            'predictor_func': lambda: get_rolling_tech_headlines(9),
            'target_col': 'rpo_yoy (%)',
        },
        {
            'source': 'PYPL Returns',
            'metric': 'Billings YoY Growth Rate (%)',
            'optimal window': 90,
            'optimal lag': 77,
            'hy estimator': -0.4958,
            'predictor_func': lambda: get_rolling_pypl(90),
            'target_col': 'billings_yoy (%)',
        },
        {
            'source': 'PYPL Returns',
            'metric': 'Large Customer QoQ Growth Rate (%)',
            'optimal window': 90,
            'optimal lag': 20,
            'hy estimator': -0.4396,
            'predictor_func': lambda: get_rolling_pypl(90),
            'target_col': 'large_customer_qoq (%)',
        },
    ]

    backtest_results = []
    detailed_oos_records = {}
    detailed_rows = []
    latest_dates = []

    for item in backtest_pairs:
        source_name = item['source']
        metric_name = item['metric']
        window_days = item['optimal window']
        lag_days = item['optimal lag']
        hy_val = item['hy estimator']
        target_col = item['target_col']

        if target_col not in df_metrics_idx.columns:
            log.warning(f"Target column {target_col} missing in quarterly metrics. Skipping {metric_name}.")
            continue

        s_pred = item['predictor_func']()
        s_target = df_metrics_idx[target_col].dropna()

        metric_info = metric_col_map.get(metric_name, {'col': target_col, 'color': '#3182bd', 'type': 'pct'})

        try:
            res, df_detail = run_walk_forward_backtest(s_pred, s_target, lag_days=lag_days, initial_train_size=10)
            pair_key = f"{source_name} -> {metric_name}"
            detailed_oos_records[pair_key] = (df_detail, metric_info)

            # Accumulate detailed rows for summary CSV
            for _, r in df_detail.iterrows():
                detailed_rows.append({
                    'source': source_name,
                    'metric': metric_name,
                    'pair_name': pair_key,
                    'date': r['date'],
                    'actual': r['actual'],
                    'predicted': r['predicted'],
                    'prior_actual': r['prior_actual'],
                    'actual_change': r['actual_change'],
                    'pred_change': r['pred_change'],
                    'correct_direction': r['correct_direction'],
                    'error': r['error'],
                })

            backtest_results.append({
                'source': source_name,
                'metric': metric_name,
                'optimal window': window_days,
                'optimal lag': lag_days,
                'hy estimator': hy_val,
                'mape (%)': round(res['mape'], 2),
                'rmse': round(res['rmse'], 2),
                'directional_hit_rate (%)': round(res['directional_hit_rate'], 2),
                'avg_risk_to_reward_ratio': round(res['avg_risk_to_reward'], 2),
                'oos_quarters_tested': res['n_oos_quarters'],
            })
            if res['latest_date_tested']:
                latest_dates.append(res['latest_date_tested'])
        except Exception as e:
            log.error(f"Error backtesting {source_name} -> {metric_name}: {e}")

    df_summary = pd.DataFrame(backtest_results)
    if not df_summary.empty:
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = SUMMARIES_DIR / 'backtesting_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        log.info(f"Saved updated backtest summary to {summary_path.name}.")

    if detailed_rows:
        df_all_details = pd.DataFrame(detailed_rows)
        details_path = SUMMARIES_DIR / 'backtest_detailed_oos.csv'
        df_all_details.to_csv(details_path, index=False)
        log.info(f"Saved detailed OOS backtest records to {details_path.name}.")

    # Generate and save walkforward_backtest.png
    if detailed_oos_records:
        plot_backtest_trajectories(detailed_oos_records, df_summary, RESULTS_DIR / 'walkforward_backtest.png')

    log.info(f"Backtesting completed. Latest quarter backtested: {latest_qtr_label}.")
    return df_summary, latest_qtr_label, detailed_oos_records


if __name__ == "__main__":
    df_res, latest_q, _ = run()
    print("\nBacktest Summary Table:")
    print(df_res.to_string())
    print(f"\nLatest Quarter Backtested: {latest_q}")
