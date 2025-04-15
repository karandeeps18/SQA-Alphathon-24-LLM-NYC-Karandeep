from AlgorithmImports import *
import numpy as np

class FamaFrenchWithPredictionAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)    
        self.SetEndDate(2019, 12, 31)   
        self.SetCash(1000000000)           

        # Tech stocks only (S&P 500 members)
        self.tech_symbols = [self.AddEquity(symbol).Symbol for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "FB", "AMZN", "NFLX"]]

        # Diversified stocks from the S&P 500 
        self.diverse_symbols = [self.AddEquity(symbol).Symbol for symbol in ["KO", "JNJ", "GS", "MS", "XOM", "GE", "PEP", "BA", "PG", "VZ"]]

        # Set up slippage and transaction fees (Interactive Brokers)
        constant_slippage = ConstantSlippageModel(0.0002)  # 0.02% slippage per transaction
        for symbol in self.tech_symbols + self.diverse_symbols:
            security = self.Securities[symbol]
            security.SetSlippageModel(constant_slippage)
            security.SetFeeModel(InteractiveBrokersFeeModel())

        # Placeholder for Fama-French 4 factors and predictions
        self.fama_french_factors = {}
        self.predictions = {}

        # Initialize Fama-French factors
        self.InitializeFamaFrenchFactors()

        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(self.tech_symbols[0]), self.TimeRules.AfterMarketOpen(self.tech_symbols[0], 10), self.Rebalance)

        # Load predictions (for tech stocks only)
        self.LoadPredictions()

    def InitializeFamaFrenchFactors(self):
        # Mocked Fama-French factors (replace this with actual factor data loading logic)
        # Example: Factors data would have columns like 'MKT', 'SMB', 'HML', 'MOM'
        self.fama_french_factors = {
            'MKT': 0.03,   # Market Risk Premium
            'SMB': 0.02,   # Small Minus Big
            'HML': 0.015,  # High Minus Low (Value)
            'MOM': 0.01    # Momentum
        }

    def LoadPredictions(self):
        # Load predictions for tech stocks from the object store
        if not self.ObjectStore.ContainsKey("cnn_predictions_with_date.csv"):
            self.Debug("Predictions file not found in ObjectStore.")
            return
        
        predictions_csv = self.ObjectStore.Read("cnn_predictions_with_date.csv")
        if not predictions_csv:
            self.Debug("Failed to load predictions from ObjectStore.")
            return
        
        # Load predictions into a dictionary with date and prediction value
        data = predictions_csv.splitlines()
        for line in data[1:]:
            date, prediction = line.split(',')
            self.predictions[date] = float(prediction)

    def GetPredictionForToday(self, symbol):
        # Get prediction for today for the specific tech symbol
        today_str = self.Time.strftime('%Y-%m-%d')
        return self.predictions.get(today_str, None)

    def Rebalance(self):
        # Rebalance based on Fama-French factors and predictions
        total_weight = 0
        tech_weights = {}
        factor_weights = {}

        # Calculate the tech stock weights based on predictions
        for symbol in self.tech_symbols:
            if not self.Securities[symbol].Price:
                continue
            
            prediction = self.GetPredictionForToday(symbol)
            if prediction is not None and prediction > 0.95:
                # Use prediction as a factor
                tech_weights[symbol] = 0.1  # Assign 10% weight to each tech stock with a prediction signal
                total_weight += 0.1

        # Allocate the rest of the portfolio based on Fama-French factors (market, size, value, momentum)
        remaining_weight = 1 - total_weight

        if remaining_weight > 0:
            factor_stock_weights = {
                'KO': 0.1 * remaining_weight,   # Example stock for market risk premium
                'JNJ': 0.15 * remaining_weight,  # Example stock for size factor (SMB)
                'GS': 0.2 * remaining_weight,    # Example stock for value factor (HML)
                'XOM': 0.1 * remaining_weight,   # Example stock for momentum (MOM)
                'PEP': 0.1 * remaining_weight,   # Remaining diversified stocks
                'PG': 0.1 * remaining_weight,
                'VZ': 0.1 * remaining_weight,
                'GE': 0.15 * remaining_weight
            }

            # Set holdings for tech stocks
            for symbol, weight in tech_weights.items():
                if weight > 0:
                    self.SetHoldings(symbol, weight)

            # Set holdings for diversified stocks based on Fama-French
            for symbol, weight in factor_stock_weights.items():
                if weight > 0:
                    self.SetHoldings(symbol, weight)

    def OnData(self, data):
        pass

    def OnEndOfDay(self):
        # Plot portfolio metrics
        self.Plot("Portfolio Value", "Total Value", self.Portfolio.TotalPortfolioValue)
