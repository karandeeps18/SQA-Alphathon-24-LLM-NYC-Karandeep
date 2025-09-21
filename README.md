# SQA ALPHATHON 2024
**Multimodal Model Approach to Generate Alpha Using FinBERT**  
**Question 2: “Use LLM to outperform S&P 500”**

## Overview
This repository demonstrates a **multimodal** trading strategy aiming to outperform the S&P 500 by identifying **extreme price movements** (±4%) in tech stocks—particularly Tesla (TSLA). Our approach combines:

1. **Technical Analysis (TA) features** from OHLCV data  
2. **FinBERT-based sentiment embeddings** from Tiingo News  
3. **Machine Learning models**—SVM, CNN, and a Fusion Model  
4. **Fama-French 4-factor** integration for non-tech diversification  
5. **Backtesting** over multiple periods (1-year and 3-year) to compare performance

---

## Table of Contents
 
1. [Dependencies](#dependencies)  
2. [Data Sources](#data-sources)  
   - [Technical Data](#technical-data)  
   - [FinBERT Embeddings & News](#finbert-embeddings--news)  
3. [Core Modules](#core-modules)  
   - [SVM for Technical Data (`svm_ta.py`)](#svm-for-technical-data)  
   - [CNN for Sentiment (`cnn_tiingo.py`)](#cnn-for-sentiment)  
   - [Fusion Model (`fusion_model.py`)](#fusion-model)  
   - [Backtest & Trading Algorithm (`trading_algo_backtest.py`)](#backtest--trading-algorithm)  
4. [Backtest Results](#backtest-results)  
   - [1-Year Performance](#1-year-performance)  
   - [3-Year Performance](#3-year-performance)  
5. [Usage & Instructions](#usage--instructions)  
   - [Environment Setup](#environment-setup)  
   - [Running the Models](#running-the-models)  
   - [Interpreting Results](#interpreting-results)  
6. [Methodology](#methodology)  
   - [Data Preparation & Normalization](#data-preparation--normalization)  
   - [Handling Class Imbalances](#handling-class-imbalances)  
   - [Fama-French Factor Integration](#fama-french-factor-integration)  
7. [Future Enhancements](#future-enhancements)  
8. [Credits & Acknowledgements](#credits--acknowledgements)  
9. [Disclaimer](#disclaimer)

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

*The strategy **outperformed** its benchmark with a ~12.9% drawdown.* possible overfitting. We can use 
*(See [bactest_1y.pdf](./bactest_1y.pdf) for the full report)* Reduce overfitting using, purged cross validation technique, introduced by **Marcos Lopez De Prado.** 

### 3-Year Performance
From **Jan 2017 to Dec 2019**  
- **CAGR**: ~5.8%  
- **Drawdown**: 23.5%  
- **Sharpe Ratio**: 0.2  
- **Sortino Ratio**: 0.2  
- **Turnover**: 1%  
- **Information Ratio**: -0.9  

*Over a longer period, the strategy struggled to beat the benchmark, indicating **period-specific** alpha.*  
*(See [3y_backtest.pdf](./3y_backtest.pdf) for the full report)* With finetuning properly better results can be obtained. 

---

## Usage & Instructions

### Environment Setup

1. **Clone** the repository:
   ```bash
   git clone https://github.com/username/alphathon-2024.git
   cd alphathon-2024

## Running the Models

### Train SVM (technical)
- **Run `svm_ta.py`** (in a QuantBook or local environment).
- Outputs stored in **ObjectStore** as `svm_probabilities_new.csv`.

### Train CNN (sentiment)
- **Run `cnn_tiingo.py`** to generate **FinBERT** embeddings and train a CNN.
- Saves **`cnn_predictions_with_date.csv`** in ObjectStore.

### Fusion
- **Execute `fusion_model.py`** to combine **SVM + CNN** probabilities.

### Backtest
- Finally, **run `trading_algo_backtest.py`** to see performance vs. a benchmark.

---

## Interpreting Results

- **Log output** includes classification reports, accuracy, and precision/recall for minority classes.
- The **QuantConnect** backtest summary reveals key portfolio metrics (CAGR, drawdown, Sharpe, etc.).
- Compare **short-term vs. long-term** performance to gauge overfitting or time-dependent alpha.

---

## Methodology

### Data Preparation & Normalization
- **Price-based features** are scaled by **prior-day close**; volume & index data by `pct_change()`.
- **FinBERT embeddings** come from the **description** fields in Tiingo News (tokenized up to 512 tokens).

### Handling Class Imbalances
- Threshold **±4%** for large moves ⇒ fewer positive samples.
- **SVM `class_weight`** & careful **threshold tuning** mitigate skewed data distribution.

### Fama-French Factor Integration
- The “**fifth factor**” is the ML model’s probability for **tech stocks**.
- **Non-tech exposure** follows **MKT, SMB, HML, MOM** factors to diversify.
- **Daily rebalancing** with constraints on position sizing (e.g., max **10%** per strong tech signal).

---

## Future Enhancements

1. **Extended Data**: Include **2020–2022** data to test pandemic/volatility resilience.  
2. **Advanced Sampling**: Use **SMOTE** or other synthetic methods for extreme-move minority oversampling.  
3. **Transformer Architectures**: Explore advanced fusion (e.g., **BERT-based** sequence models).  
4. **Dynamic Thresholding**: Adapt thresholds in real-time based on volatility regimes.

---

## Credits & Acknowledgements

- **Research Inspiration**:  
  - “*Prebit - A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin*” by Yanzhao Zou & Dorien Herremans  
- **Academic Guidance**: Professors *Dimitry Udler* and Professor *N. K. Chidambaran* (Fordham University)  
- **QuantConnect**: Lean environment, Data feeds, and ObjectStore for backtesting

---

## Disclaimer

All materials and code herein are provided for **research and educational purposes** only.  
This does **not** constitute financial advice. **Trading involves risk**; past performance does **not** guarantee future results.

[COMPLETE REPORT AND BACKTEST](SQA ALPHATHON 2024.pdf)
