# SQA ALPHATHON 2024
**Multimodal Model Approach to Generate Alpha Using FinBERT**  
**Question 2: “Use LLM to outperform S&P 500”**

:contentReference[oaicite:0]{index=0}  
:contentReference[oaicite:1]{index=1}  

## Overview
This repository demonstrates a **multimodal** trading strategy aiming to outperform the S&P 500 by identifying **extreme price movements** (±4%) in tech stocks—particularly Tesla (TSLA). Our approach combines:

1. **Technical Analysis (TA) features** from OHLCV data  
2. **FinBERT-based sentiment embeddings** from Tiingo News  
3. **Machine Learning models**—SVM, CNN, and a Fusion Model  
4. **Fama-French 4-factor** integration for non-tech diversification  
5. **Backtesting** over multiple periods (1-year and 3-year) to compare performance

---

## Table of Contents

1. [Repository Structure](#repository-structure)  
2. [Dependencies](#dependencies)  
3. [Data Sources](#data-sources)  
   - [Technical Data](#technical-data)  
   - [FinBERT Embeddings & News](#finbert-embeddings--news)  
4. [Core Modules](#core-modules)  
   - [SVM for Technical Data (`svm_ta.py`)](#svm-for-technical-data)  
   - [CNN for Sentiment (`cnn_tiingo.py`)](#cnn-for-sentiment)  
   - [Fusion Model (`fusion_model.py`)](#fusion-model)  
   - [Backtest & Trading Algorithm (`trading_algo_backtest.py`)](#backtest--trading-algorithm)  
5. [Backtest Results](#backtest-results)  
   - [1-Year Performance](#1-year-performance)  
   - [3-Year Performance](#3-year-performance)  
6. [Usage & Instructions](#usage--instructions)  
   - [Environment Setup](#environment-setup)  
   - [Running the Models](#running-the-models)  
   - [Interpreting Results](#interpreting-results)  
7. [Methodology](#methodology)  
   - [Data Preparation & Normalization](#data-preparation--normalization)  
   - [Handling Class Imbalances](#handling-class-imbalances)  
   - [Fama-French Factor Integration](#fama-french-factor-integration)  
8. [Future Enhancements](#future-enhancements)  
9. [Credits & Acknowledgements](#credits--acknowledgements)  
10. [Disclaimer](#disclaimer)

---

## Repository Structure
.

├── 1yearbacktes.pdf

├── 3y_backtest.pdf

├── bactest_1y.pdf

├── cnn_tiingo.py

├── fusion_model.py

├── SQA ALPHATHON 2024.pdf

├── svm_ta.py

└── trading_algo_backtest.py



---

## Dependencies

This project relies on the following key libraries and frameworks:
- **Python 3.8+**
- **[NumPy](https://pypi.org/project/numpy/)**
- **[pandas](https://pypi.org/project/pandas/)**
- **[scikit-learn](https://pypi.org/project/scikit-learn/)**
- **[TensorFlow](https://pypi.org/project/tensorflow/)** (for CNN and FinBERT model integration)
- **[transformers](https://pypi.org/project/transformers/)** (for FinBERT)
- **[matplotlib](https://pypi.org/project/matplotlib/)**
- **[seaborn](https://pypi.org/project/seaborn/)**
- **QuantConnect/Lean CLI or Research** environment (if running on QuantConnect)
- Any additional libraries specified in `requirements.txt`

---

## Data Sources

### Technical Data
- **Tesla (TSLA)** & **SPY** OHLCV data for the period **2019-01-01** to **2019-12-31** (can extend to 3+ years for broader tests).  
- **Engineered indicators**: SMA, EMA (12, 26, exponential decay), Bollinger Bands, MACD, High-Low spreads.  
- **Feature normalization** uses a custom scaling (percentage changes, etc.).

### FinBERT Embeddings & News
- **Tiingo News** referencing Tesla.  
- **FinBERT** for sentiment classification & embedding generation (768-dimensional).  
- **CNN** processes these embeddings to yield sentiment-driven signals.

---

## Core Modules

### SVM for Technical Data
File: `svm_ta.py`  
- Trains an **SVM** with an **RBF kernel** on normalized TA features.  
- **Class weighting** for rare (±4%) moves.  
- Outputs probability estimates (`svm_probabilities_new.csv`).

### CNN for Sentiment
File: `cnn_tiingo.py`  
- Cleans news text, applies **FinBERT** to generate embeddings.  
- A **parallel CNN** captures textual patterns from the embeddings.  
- Saves predictions as `cnn_predictions_with_date.csv`.

### Fusion Model
File: `fusion_model.py`  
- Merges the **SVM** (technical) and **CNN** (sentiment) probabilities.  
- Final classifier (SVM) predicts if next-day return > 4%.  
- Generates a fused output stored in ObjectStore.

### Backtest & Trading Algorithm
File: `trading_algo_backtest.py`  
- **QuantConnect** algorithm combining:  
  1. **ML-based “5th factor”** for tech picks (e.g., TSLA, AAPL)  
  2. **Fama-French 4-Factor** for non-tech diversification  
- Applies daily rebalancing, transaction fees, slippage models, and logs performance.

---

## Backtest Results

### 1-Year Performance
From **Jan 2019 to Dec 2019**  
- **CAGR**: ~44.0%  
- **Drawdown**: 12.9%  
- **Sharpe Ratio**: 1.9  
- **Sortino Ratio**: 2.2  
- **Turnover**: 1%  
- **Information Ratio**: 1.1  

*The strategy **outperformed** its benchmark with a ~12.9% drawdown.*  
*(See [bactest_1y.pdf](./backtests/bactest_1y.pdf) for the full report)*

### 3-Year Performance
From **Jan 2017 to Dec 2019**  
- **CAGR**: ~5.8%  
- **Drawdown**: 23.5%  
- **Sharpe Ratio**: 0.2  
- **Sortino Ratio**: 0.2  
- **Turnover**: 1%  
- **Information Ratio**: -0.9  

*Over a longer period, the strategy struggled to beat the benchmark, indicating **period-specific** alpha.*  
*(See [3y_backtest.pdf](./backtests/3y_backtest.pdf) for the full report)*

---

## Usage & Instructions

### Environment Setup

1. **Clone** the repository:
   ```bash
   git clone https://github.com/username/alphathon-2024.git
   cd alphathon-2024
