# Overview
Datadog (NASDAQ:DDOG) is a widely popular platform providing cloud observability and security capabilities to other businesses. My project proposes **three alternative data sources** that could be used to predict the key financial and operating metrics reported by Datadog: PayPal (NASDAQ:PYPL) stock returns, thematic sentiment score of technology news headlines, and the frequency of Datadog's product releases. A **quantitative analysis** of the extent to which the alternative datasets can act as leading indicators to the quarterly metrics is conducted to identify potentially effective trading strategies. The trading strategies are out-of-sample (OOS) **backtested**. Further details of this study are contained in the short report from August 2026, `github_Report.pdf`. [This interactive dashboard](https://ddog-revenue-nowcasting-lim-li-xuan-phy.streamlit.app/) was created to facilitate trading on the promising trading strategies identified in the study.

# 📁 Project Structure
```
ddog-revenue-nowcasting/
│
├── README.md                          # This file: Overview
│
├── DDOG_Revenue_Nowcasting_Report.pdf # Short report on data sources, methodologies, analysis, and findings
│
├── Assets/
│   └── dd_logo_v_rgb.png             # Company logo for dashboard
│
├── Data/                           # Data used for analysis
│   ├── qtrly_metrics.csv           # Key quarterly metrics reported by Datadog
│   ├── pypl_returns.csv            # PayPal stock returns
│   ├── tech_headlines.csv          # Technology news headlines
│   ├── tech_headlines_labelled.csv # Labelled examples of technology headlines for NLP model training
│   └── announcements.csv           # Press announcements by Datadog
│
├── Code/
│   ├── collect_*.py          # Gets data from online sources
│   ├── run_all_collectors.py # Automatically executes all python scripts named collect_*.py
│   ├── run_backtest.py       # Runs backtesting automatically for dashboard
│   ├── requirements.txt      # Dependencies for streamlit to deploy dashboard
│   ├── app.py                # Code for interactive dashboard 
│   ├── run_app.sh            # Uncompiled script that executes dashboard on Bash terminal
│   ├── run_app.bat           # Compiled script to execute dashboard
│   └── Notebooks/
│       ├── 01_Review_Data.ipynb       # Visualize raw data vs time
│       ├── 02_Lead_Lag_Analysis.ipynb # Analysis of lead-lad relationship between alternative data and quarterly metrics
│       ├── 03_Prediction.ipynb        # Machine-learning prediction of quarterly metrics from alternative data
│       └── 04_Backtesting.ipynb       # Walk-forward backtests of quarterly metrics' linear regression predictive model
│
└── Results/
    └── Summaries/
        ├── heatmap_lead_lag.png          # Heatmap of maximum HY estimators
        ├── hy_summary.csv                # Results for maximum HY estimator occurring where data source leads quarterly metric
        ├── heatmap_lead_lag_pearsons.png # Heatmap of maximum Pearson's correlation coefficients
        ├── pearsons_summary.csv          # Results for maximum Pearsons' coefficient occurring where data source leads quarterly metric
        ├── strongest_correlations.csv    # Combinations of data source and quarterly metric that give the highest linear regression R2 score
        ├── every_backtest_prediction.csv # Results of every backtested datapoint
        └── backtesting_summary.csv       # Overall backtesting metrics results
```

# 🏅 Acknowledgements
- Datadog's quarterly reported metrics and press releases were taken from the **Datadog website**.
- Paypal stock price data was provided by the **NASDAQ historical archive**.
- Technology news headlines were obtained from the **Consumer News and Business Channel (CNBC) website**.
- Dashboard was made using **streamlit**.
- Boilerplate code generation was AI-assisted by **Google Antigravity** and modified by me. Any errors are mine.

