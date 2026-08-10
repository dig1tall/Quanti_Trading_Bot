# 📈 Quanti Automated Crypto Trading Bot

An end-to-end quantitative machine learning platform and automated trading bot designed for **crypto perpetual futures markets**.

**Quanti** integrates market data ingestion, custom indicator feature engineering, sequence modeling via a custom **QuantiGRU** neural network, robust vector/event-driven backtesting with strategy optimization, and direct live execution on the **Bybit API**.

---

## Key Highlights

* **Automated Data Pipeline:** Fetches raw OHLCV market candles via CCXT, performs technical analysis feature generation, and prepares sliding sequence datasets.
* **QuantiGRU Architecture:** Custom PyTorch Recurrent Neural Network optimized for daily/intraday sequential classification (Long, Short, Flat).
* **Asymmetric Loss Function:** Custom `QuantiTradingLoss` combining Focal Loss with a directional cost penalty matrix to discourage high-risk opposite-direction trade misclassifications.
* **Hyperparameter Optimization:** Automated grid/random strategy optimization to identify optimal position sizing, leverage, stop-loss, take-profit, and classification confidence thresholds.
* **Live Execution Engine:** Autonomous order execution and position monitoring for Bybit USDT Perpetual Futures with real-time logging and risk controls.
* **Modular Pipeline Architecture:** Isolated engine design with clean separation of data engineering, model training, strategy evaluation, and order execution.

---

---

## Modular & Customizable Framework

Quanti is built as an extensible quantitative framework where every component can be tweaked in just a few clicks via `config.py`:

* **Data & Assets:** Switch tickers (e.g., `BTC-USDT`, `ETH-USDT`), adjust timeframes (`1d`, `1h`, `15m`), or change lookback horizons without modifying core engine logic.
* **Feature Engineering:** Add, toggle, or tune technical indicators (RSI, MACD, Bollinger Bands, ATR) in `src/features.py` with automatic alignment across scaling pipelines.
* **Model Architecture:** Easily adjust GRU hidden layer dimensions, layer depth, sequence length, or dropout rates via model parameters.
* **Risk Management:** Fine-tune long/short confidence entry thresholds, soft-exit bounds, trailing stop-losses, and take-profit targets directly in config setups.

---

## Tech Stack

* **Python 3.10+**
* **PyTorch:** Deep learning framework for QuantiGRU training, GPU acceleration, and inference.
* **CCXT:** Standardized cryptocurrency exchange trading library for market data and order placement.
* **Pandas & NumPy:** High-performance vector calculations and dataset manipulation.
* **scikit-learn:** Feature scaling (`RobustScaler`), metric computations, and confusion matrix validation.

---

## Project Structure
```
Quanti/
│── main.py                      # Global pipeline orchestrator (Data, Train, Optimize, Backtest, Live)
│── config.py                    # Centralized project configurations, paths, and hyperparameters
│── requirements.txt             # Python dependencies
│── .env                         # API key secrets (Bybit credentials)
│── .gitignore
│── README.md
│
├── src/
│   │── init.py
│   │── data_loader.py           # Historical CCXT OHLCV fetching engine
│   │── features.py             # Feature engineering & technical indicator calculation
│   │── data_preprocessing.py    # Robust scaling, sequence creation, and data splitting
│   │── dataset.py              # PyTorch Dataset & DataLoader wrappers
│   │── data_engine.py          # Data ingestion pipeline coordinator
│   │── model.py                # QuantiGRU neural network architecture definition
│   │── train.py                # Training loop with Focal + Ordinal loss and early stopping
│   │── backtest.py             # Strategy backtester and equity curve simulation
│   │── backtest_optimize.py    # Strategy parameter search and optimization
│   │── model_engine.py         # Training and backtesting coordinator
│   └── execution.py            # Live Bybit order execution loop and risk engine
│
├── reports/                     # Historical backtest outputs and execution logs
│   ├── backtest_date.html   # Interactive HTML performance dashboard
│   └── Backtest_Table_Log.md   # Obsidian-formatted markdown summary table
├── data/                       # Parquet datasets (Raw, Processed, Train, Validation)
└── models/                     # Saved PyTorch checkpoint weights (.pth)
```

---

## System Architecture & Pipeline

1. **Data Ingestion (`data_engine.py`):** Fetches historical OHLCV data from Bybit via CCXT (`data_loader.py`), computes technical indicators (`features.py`), scales features using `RobustScaler`, and constructs sequence blocks (`data_preprocessing.py`).
2. **Model Training (`train.py`):** Trains the **QuantiGRU** neural network on sequential technical features using `QuantiTradingLoss` (Focal Loss + Ordinal Penalty Matrix) to maximize Macro F1 score and minimize trade direction errors.
3. **Strategy Optimization & Backtesting (`backtest_optimize.py`, `backtest.py`):** Evaluates performance across historical datasets, tuning prediction confidence thresholds, leverage, stop-loss percentages, and take-profit targets.
4. **Live Execution (`execution.py`):** Connects to Bybit Perpetual Futures API, pulls live candle features, streams real-time model predictions, and manages automated entry, exit, and stop-loss orders.

---

## Backtesting Reports & Analytics

The framework generates execution logs and backtest analytics automatically saved in the `reports/` directory:

* **Interactive HTML Performance Reports (`.html`):** Standalone visual reports generated during backtest runs, including equity curves, drawdown analysis, trade duration distribution, and risk metrics (Sharpe ratio, Win Rate, Max Drawdown).
* **Obsidian Experiment Log (`.md`):** A structured Markdown summary table archiving test iterations, strategy parameters (thresholds, leverage, stop-loss/take-profit levels), and comparative performance metrics across validation datasets.

You can inspect the generated HTML files directly in any web browser or open the Markdown summary in **Obsidian** / GitHub to review historical experiment logs.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/dig1tall/Quanti.git](https://github.com/dig1tall/Quanti.git)
cd Quanti
```

### 2. Set Up Virtual Environment
Create and activate a standard Python virtual environment (venv):

```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (.env)
Create a .env file in the project root directory and insert your Bybit API credentials:
```dotenv
BYBIT_DEMO_API_KEY=your_bybit_api_key_here
BYBIT_DEMO_API_SECRET=your_bybit_api_secret_here
```
The live execution module (src/execution.py) reads these exact environment variables (BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET) to authenticate with Bybit USDT Perpetual Futures API.

## Usage & Execution
The entire pipeline is managed through main.py. Pipeline execution stages are controlled using configuration flags inside the main() entry point:
```python
# main.py execution flags
RUN_DATA_PIPELINE = True     # Ingest data & build datasets
RUN_MODEL_TRAINING = True    # Train QuantiGRU network
RUN_OPTIMIZATION   = True    # Tune strategy hyperparameters
RUN_BACKTESTING    = True    # Run historical backtest
RUN_LIVE           = False   # Enable automated live trading loop
```

### Run Full Pipeline
To execute the active pipeline stages:
```bash
python main.py
```

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Under this license, you are free to inspect and modify the code for personal or research use. Commercial closed-source redistribution or re-licensing without explicit permission from the original author is strictly prohibited.

## Author

**Dovgash Matvey**
- **GitHub:** [@dig1tall](https://github.com/dig1tall)
- **Email:** [matdov827@gmail.com](mailto:matdov827@gmail.com)