SQA ALPHATHON 2024
Multimodal Model Approach to Generate Alpha Using FinBERT
Question 2: “Use LLM to outperform S&P 500” - AllianceBernstien

Overview
This repository demonstrates a multimodal trading strategy aiming to outperform the S&P 500 by identifying extreme price movements (±4%) in tech stocks—particularly Tesla (TSLA). Our approach combines:

Technical Analysis (TA) features from OHLCV data.

FinBERT-based sentiment embeddings from Tiingo News.

Machine Learning models—SVM, CNN, and a Fusion Model.

Fama-French 4-factor integration for non-tech diversification.

Backtesting over multiple periods (1-year and 3-year) to compare performance.

Table of Contents
Repository Structure

Data Sources

Technical Data

FinBERT Embeddings & News

Core Modules

SVM for Technical Data (svm_ta.py)

CNN for Sentiment (cnn_tiingo.py)

Fusion Model (fusion_model.py)

Backtest & Trading Algorithm (trading_algo_backtest.py)

Backtest Results

1-Year Performance

3-Year Performance

Usage & Instructions

Environment Setup

Running the Models

Interpreting Results

Methodology

Data Preparation & Normalization

Handling Class Imbalances

Fama-French Factor Integration

Future Enhancements

Credits & Acknowledgements

Disclaimer

Repository Structure
graphql
Copy
Edit
.
├── fusion_model.py                 # Combines output probabilities from SVM & CNN
├── svm_ta.py                       # SVM classifier for technical indicators
├── cnn_tiingo.py                   # CNN model leveraging FinBERT embeddings
├── trading_algo_backtest.py        # Fama-French + ML-based trading logic
├── data/
│   ├── technical/                  # TSLA & SPY OHLCV data + indicators
│   ├── news/                       # Tiingo News data
│   └── processed/                  # Preprocessed / merged datasets
├── backtests/
│   ├── results/                    # Logs, CSV output
│   ├── plots/                      # Equity curves, monthly returns
│   ├── bactest_1y.pdf             # 1-year backtest report
│   └── 3y_backtest.pdf            # 3-year backtest report
└── README.md
Data Sources
Technical Data
Tesla (TSLA) & SPY OHLCV data for the period 2019-01-01 to 2019-12-31 (can extend to 3+ years for broader tests).

Engineered indicators: SMA, EMA (12, 26, exponential decay), Bollinger Bands, MACD, High-Low spreads.

Feature normalization uses a custom scaling (percentage changes, etc.).

FinBERT Embeddings & News
Tiingo News referencing Tesla.

FinBERT for sentiment classification & embedding generation (768-dimensional).

CNN processes these embeddings to yield sentiment-driven signals.

Core Modules
SVM for Technical Data
File: svm_ta.py

Trains an SVM with an RBF kernel on normalized TA features.

Class weighting for rare (±4%) moves.

Outputs probability estimates (svm_probabilities_new.csv).

CNN for Sentiment
File: cnn_tiingo.py

Cleans news text, applies FinBERT to generate embeddings.

A parallel CNN captures textual patterns from the embeddings.

Saves predictions as cnn_predictions_with_date.csv.

Fusion Model
File: fusion_model.py

Merges the SVM (technical) and CNN (sentiment) probabilities.

Final classifier (SVM) predicts if next-day return > 4%.

Generates a fused output stored in ObjectStore.

Backtest & Trading Algorithm
File: trading_algo_backtest.py

QuantConnect algorithm combining:

ML-based “5th factor” for tech picks (e.g., TSLA, AAPL).

Fama-French 4-Factor for non-tech diversification.

Applies daily rebalancing, transaction fees, slippage models, and logs performance.

Backtest Results
1-Year Performance
From Jan 2019 to Dec 2019

CAGR: ~44.0%

Drawdown: 12.9%

Sharpe Ratio: 1.9

Sortino Ratio: 2.2

Turnover: 1%

Information Ratio: 1.1

The strategy outperformed its benchmark with a ~12.9% drawdown.

(See bactest_1y.pdf for the full report, including monthly returns, equity curve, rolling Sharpe ratio, etc.)

3-Year Performance
From Jan 2017 to Dec 2019

CAGR: ~5.8%

Drawdown: 23.5%

Sharpe Ratio: 0.2

Sortino Ratio: 0.2

Turnover: 1%

Information Ratio: -0.9

Over a longer period, the strategy struggled to beat the benchmark, suggesting period-specific alpha.

(See 3y_backtest.pdf for the full report.)

Usage & Instructions
Environment Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/username/alphathon-2024.git
cd alphathon-2024
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
QuantConnect / Lean:

Upload these scripts into your QuantConnect Project or use Lean CLI locally.

Running the Models
Train SVM (technical):

Run svm_ta.py (in a QuantBook or local environment).

Outputs stored in ObjectStore (svm_probabilities_new.csv).

Train CNN (sentiment):

Run cnn_tiingo.py to generate FinBERT embeddings and train a CNN.

Saves cnn_predictions_with_date.csv in ObjectStore.

Fusion:

Execute fusion_model.py to combine SVM + CNN probabilities.

Backtest:

Finally, run trading_algo_backtest.py to see performance vs. a benchmark.

Interpreting Results
Log output includes classification reports, accuracy, precision/recall for minority classes.

The QuantConnect backtest summary reveals portfolio metrics (CAGR, drawdown, Sharpe, etc.).

Compare short-term vs. long-term performance to gauge overfitting or time-dependent alpha.

Methodology
Data Preparation & Normalization
Price-based features scaled by prior-day close; volume & index data by pct_change().

FinBERT embeddings from description fields in Tiingo News (tokenized up to 512 tokens).

Handling Class Imbalances
Threshold ±4% for large moves leads to fewer positive samples.

SVM class_weight & careful threshold tuning mitigate skewed data distribution.

Fama-French Factor Integration
The “fifth factor” is the ML model’s probability for tech stocks.

Non-tech exposure follows MKT, SMB, HML, MOM factors to diversify.

Daily rebalancing with constraints on position sizing (e.g., max 10% per strong tech signal).

Future Enhancements
Extended Data: Include 2020–2022 data to test pandemic/volatility resilience.

Advanced Sampling: Use SMOTE or other synthetic methods for extreme-move minority oversampling.

Transformer Architectures: Explore advanced fusion approaches (e.g., BERT-based sequence models).

Dynamic Thresholding: Adapt thresholds in real-time based on volatility regimes.

Credits & Acknowledgements
Research Inspiration:

“Prebit - A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin” by Yanzhao Zou & Dorien Herremans.

Academic Guidance: Professors Dimitry Udler and N. K. Chidambaran (Fordham University).

QuantConnect: Lean environment, Data feeds, and ObjectStore for backtesting.

Disclaimer
All materials and code herein are provided for research and educational purposes only. This does not constitute financial advice. Trading involves risk; past performance does not guarantee future results.
