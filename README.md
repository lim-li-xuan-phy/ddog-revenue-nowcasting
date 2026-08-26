# *Project completed. README writing is in progress...* 👷‍♂️🚧

# Overview
Datadog (NASDAQ:DDOG) is a widely popular platform providing cloud observability and security capabilities to other businesses. My project proposes **three alternative data sources** that could be used to predict the key financial and operating metrics reported by Datadog: PayPal (NASDAQ:PYPL) stock returns, thematic sentiment score of technology news headlines, and the frequency of Datadog's product releases. A **quantitative analysis** of the extent to which the alternative datasets can act as leading indicators to the quarterly metrics is conducted to identify potentially effective trading strategies. The trading strategies are out-of-sample (OOS) **backtested**. Further details of this study are contained in the report written in August 2026, `DDOG_Revenue_Nowcasting_Report.pdf`. [This interactive dashboard](https://ddog-revenue-nowcasting-lim-li-xuan-phy.streamlit.app/) was created to facilitate trading on the promising trading strategies identified in the study.

# Table of Contents
- [Project structure](#project-structure)
- [Reproduce the analysis](#reproduce-the-analysis)
- [Acknowledgements](#acknowledgements)

# Project Structure
```
ddog-revenue-nowcasting/
│
├── README.md                       # This file: Overview
│
├── DDOG_Revenue_Nowcasting_Report.pdf # Short report on data sources, methodologies, analysis, and findings
│
├── Assets/
│   └── dd_logo_v_rgb.png           # Company logo for dashboard
│
├── Data/                           # Data used for analysis
│   ├── qtrly_metrics.csv           # Key quarterly metrics reported by Datadog
│   ├── pypl_returns.csv            # PayPal stock returns
│   ├── tech_headlines.csv          # Technology news headlines
│   ├── tech_headlines_labelled.csv # Labelled examples of technology headlines for NLP model training
│   └── announcements.csv           # Press announcements by Datadog
│
├── Code/
│   ├── collect_*.py            # Gets data from online sources
│   ├── run_all_collectors.py   # Automatically executes all python scripts named collect_*.py
│   ├── run_backtest.py         # Runs backtesting automatically for dashboard
│   ├── app.py                  # Code for interactive dashboard 
│   ├── run_app.sh              # Uncompiled script that executes dashboard on Bash terminal
│   ├── run_app.bat             # Compiled script to execute dashboard
│   └── Notebooks/
│       ├── Review_Data.ipynb       # Visualize raw data vs time
│       ├── Lead_Lag_Analysis.ipynb # Analysis of lead-lad relationship between alternative data and quarterly metrics
│       ├── Prediction.ipynb        # Machine-learning prediction of quarterly metrics from alternative data
│       └── Backtesting.ipynb       # Walk-forward backtests of quarterly metrics' linear regression predictive model
│
└── Results/
    └── Summaries/
        ├── heatmap_lead_lag.png
        ├── hy_summary.csv
        ├── heatmap_lead_lag_pearsons.png
        ├── pearsons_summary.csv
        ├── strongest_correlations.csv
        ├── every_backtest_prediction.csv
        └── backtesting_summary.csv
```

# Reproduce the analysis


# Acknowledgements
- Dashboard was made using streamlit.
- Boilerplate code generation was AI-assisted by Google Antigravity.