# %% [markdown]
# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
# <hr>

# %% [markdown]
# # Training SVM using Technical Data 

# %%
from QuantConnect import *
from QuantConnect.Research import QuantBook
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.utils import resample 
import seaborn as sns 
from QuantConnect.Data.UniverseSelection import *
import math
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from datetime import datetime


qb = QuantBook()
data = pd.DataFrame()

# %%
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)

# %%
spy_symbol = qb.AddEquity("SPY").Symbol 
tsla_symbol = qb.AddEquity("TSLA").Symbol

# %%
spy_history = qb.History(spy_symbol, start_date, end_date, Resolution.Daily)
tsla_history = qb.History(tsla_symbol, start_date, end_date, Resolution.Daily)

# %%
spy_history_reset = spy_history.reset_index(level=['symbol', 'time'])
tsla_history_reset = tsla_history.reset_index(level=['symbol', 'time'])

# %%
spy_history_reset = spy_history_reset.drop(columns='symbol')
tsla_history_reset = tsla_history_reset.drop(columns='symbol')

# %%
combined_data = pd.merge(spy_history_reset, tsla_history_reset, on='time', suffixes=('_spy', '_tsla'))
combined_data.set_index('time', inplace=True)

# %%
combined_data

# %% [markdown]
# ## Technical Indicators
# 
# 1. 7-day SMA 
# 2. 21-day SMA 
# 3. EMA with 0.67 decay factor 
# 4. 12-Day EMA 
# 5. 26-day EMA 

# %%
# Simple Moving Averages (SMA)
combined_data['tsla_SMA_7'] = combined_data['close_tsla'].rolling(window=7).mean()
combined_data['tsla_SMA_21'] = combined_data['close_tsla'].rolling(window=21).mean()

    # Exponential Moving Averages (EMA)
combined_data['tsla_EMA_decay'] = combined_data['close_tsla'].ewm(alpha=0.67, adjust=False).mean()
combined_data['tsla_EMA_12'] = combined_data['close_tsla'].ewm(span=12, adjust=False).mean()
combined_data['tsla_EMA_26'] = combined_data['close_tsla'].ewm(span=26, adjust=False).mean()

    # Standard Deviation and Bollinger Bands
combined_data['tsla_20d_stdDev'] = combined_data['close_tsla'].rolling(window=20).std()
combined_data['tsla_Upper_BB'] = combined_data['tsla_SMA_21'] + (combined_data['tsla_20d_stdDev'] * 2)
combined_data['tsla_Lower_BB'] = combined_data['tsla_SMA_21'] - (combined_data['tsla_20d_stdDev'] * 2)

    # High-Low Spread
combined_data['tsla_High_Low_Spread'] = combined_data['high_tsla'] - combined_data['low_tsla']
combined_data.describe()


# %%
combined_data['MACD'] = combined_data['tsla_EMA_12'] - combined_data['tsla_EMA_26'] 

# %%
combined_data.dropna()

# %%
# Plotting TA
plt.figure(figsize=(10, 5))
plt.plot(combined_data['close_tsla'], label='tsla Close Price')
plt.plot(combined_data['tsla_SMA_7'], label='tsla SMA 7')
plt.plot(combined_data['tsla_SMA_21'], label='tsla SMA 21')
plt.plot(combined_data['tsla_EMA_decay'], label='tsla EMA Decay')
plt.plot(combined_data['tsla_EMA_12'], label='tsla EMA 12')
plt.plot(combined_data['tsla_EMA_26'], label='tsla EMA 26')
plt.plot(combined_data['tsla_Upper_BB'], label='tsla EMA 26')
plt.plot(combined_data['tsla_Lower_BB'], label='tsla EMA 26')
plt.title('TSLA Technical Indicators')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()



# %% [markdown]
# # Feature Engineering

# %%
features = combined_data[[ 'close_tsla',	'high_tsla',	'low_tsla',	'open_tsla',	'volume_tsla',
    'tsla_SMA_7', 'tsla_SMA_21', 'tsla_EMA_decay', 'tsla_EMA_12', 'tsla_EMA_26', 'MACD',
    'tsla_20d_stdDev', 'tsla_Upper_BB', 'tsla_Lower_BB', 'tsla_High_Low_Spread', 'close_spy', 'volume_spy']]

# %%
combined_data

# %%
combined_data['Target'] = (combined_data['close_tsla'].shift(-1) > combined_data['close_tsla']).astype(int) # target variable 

# %%
correlation_matrix = combined_data.corr(method='pearson')

#Heat map
plt.figure(figsize=(16, 12))  
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 10}, cbar_kws={'label': 'Pearson Correlation'}) 
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)  
plt.yticks(fontsize=12)  
plt.tight_layout()  
plt.show()


# %% [markdown]
# # Normalization 
# 
# 1. Using Standard Scaler from scikit-learn mu=0 and Std=1
# 2. Using Min-Max Scaler to define upper lower bounds
# 3. Custom Scaling 

# %%
scaler = StandardScaler()
standard_scaler = scaler.fit_transform(features)
standard_scaler = pd.DataFrame(standard_scaler, columns=features.columns)
standard_scaler.head()

# %%
standard_scaler.tail()

# %%
scaler_mm = MinMaxScaler()
mmf = scaler_mm.fit_transform(features)
mmf = pd.DataFrame(mmf, columns=features.columns)
mmf.tail()

# %% [markdown]
# ### Custom Scalers as the Price and Volume needs to Scaled Differently

# %%
# features that are directly related to the price should be noramalized as percentage change from the previous day's volume 
scaled_data = pd.DataFrame()

price_features = ['close_tsla',	'high_tsla', 'low_tsla', 'open_tsla', 'tsla_SMA_7',	'tsla_SMA_21', 'tsla_EMA_decay', 'tsla_EMA_12', 'tsla_EMA_26']

for features in price_features:

    scaled_data[features + '_scaled'] = combined_data[features] / combined_data['close_tsla'].shift(1) - 1 

# %%
# For features that are realted to volume should be scaled as thier own percent change
volume_features = ['volume_tsla', 'close_spy', 'volume_spy']

for features in volume_features:
    scaled_data[features + '_scaled'] = combined_data[features].pct_change()

# %%
# For the features that are Momentum Indicators must be normalized with the Price suggesting overbought and oversold conditions (momentum based)
other_features = ['MACD', 'tsla_20d_stdDev', 'tsla_High_Low_Spread']

for features in other_features:
    scaled_data[features + '_scaled'] = combined_data[features] / combined_data['close_tsla'].shift(1) - 1 

# %%
scaled_data.dropna()

# %% [markdown]
# ### Class Label 
# 1. Now as we have to predict the extreem price movements of tsla using a hybrid multimodal approach we need to create a binary class for extreem movements. 
# 2. For which we need to set our threshold theta. 
# 3. For which we need to examine the distribution of %change to identify the suitable threshold

# %%
scaled_data['tsla_returns'] = combined_data['close_tsla'].pct_change()
summary_stats=scaled_data['tsla_returns'].describe()
print(summary_stats)

# %%
n = len(scaled_data)
b = math.log2(n)
bins = round(b) # Following sturges  for selecting bins >> law Bins = Log base 2 (n)
plt.figure(figsize=(10, 6)) 
plt.hist(scaled_data['tsla_returns'], bins=bins, edgecolor='black', color='skyblue')  
plt.title('Distribution of TSLA Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Creating Class Labels for SVM 
# 
# 1. We will have +-2% threshold which will serve as a flag smaller price movement 
# 2. +-4% for large movements
# 3. As the data is right skewed or biased we need to create a imbalanced classification
# 
# Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. This results in models that have poor predictive performance, *specifically for the minority class.* This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class. 
# 
# As the minority class is at most important for higher gains and risk management for down-side risk we need to define a *class distribution ratio*. This can be used for anamoly detection
# 
# 

# %%
low_theta = 0.02
high_theta = 0.03

scaled_data['Class_labels'] = np.select(
    [
        scaled_data['tsla_returns'].abs() < low_theta, #Normal Movements 
        scaled_data['tsla_returns'].abs() >= high_theta, #high movements 

        
    ], 
    [0,1]
)

class_0 = scaled_data[scaled_data['Class_labels'] == 0] 
class_1 = scaled_data[scaled_data['Class_labels'] == 1] 

# %%
correlation_matrix = scaled_data.corr(method='pearson')

# heatmap
plt.figure(figsize=(16, 12))  
heatmap = sns.heatmap(correlation_matrix, 
                      annot=True, 
                      cmap='coolwarm', 
                      fmt='.2f',        
                      linewidths=0.5,   
                      annot_kws={"size": 10},  
                      cbar_kws={'label': 'Pearson Correlation'})  

# Set titles and labels
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)  
plt.yticks(fontsize=12)  
plt.tight_layout()  

# Show the heatmap
plt.show()


# %% [markdown]
# #### tuning instructions for Different level of Risks. A class ratio of Institutional level risk can be directly used as ratios along with the class distribution to develop strategies for different risk profiles
# 
# 1. tune n1,n0 to adjust according to the market (if economic indicators suggest downturn scale down the n1 and n0, or other way around you can oversample the 2 by creating synthetic data by using SMOTE, (Synthetic Minority Over-sampling technique. SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b. A general downside of the approach is that synthetic examples are created without considering the majority class, possibly resulting in ambiguous examples if there is a strong overlap for the classes)

# %%
classes = np.array([0, 1])

# computing class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=scaled_data['Class_labels'])
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights: ", class_weight_dict)

# %% [markdown]
# # Selecting features 
# 
# 

# %%
scaled_data.dropna()
print(scaled_data.columns)

# %%
selected_features = [ 'close_tsla_scaled', 'tsla_SMA_7_scaled', 'tsla_EMA_decay_scaled', 'tsla_EMA_26_scaled',	'volume_tsla_scaled',	'close_spy_scaled',	'volume_spy_scaled', 'MACD_scaled',	'tsla_20d_stdDev_scaled', 'tsla_High_Low_Spread_scaled']

# %%
X = scaled_data[selected_features] 
y = scaled_data['Class_labels']  

X = X.dropna()
y = y.loc[X.index] 

#(70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Further spliting the remaining data (X_temp) into validation and test sets (50/50)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# shape of the splits
print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

# %% [markdown]
# ## Training SVM on TA Data 

# %%
from sklearn.svm import SVC

clf = SVC(kernel='rbf', class_weight={0: 0.57, 1: 4.04}, C=1, gamma='scale')
clf = SVC(probability=True)

# Train the model on the training set
clf.fit(X_train, y_train)

# Validation set
y_val_pred = clf.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))

# test set
y_test_pred = clf.predict(X_test)
print("Test Set Performance:")
print(classification_report(y_test, y_test_pred))

# %%
probabilities = clf.predict_proba(X_test)
probabilities_df = pd.DataFrame(probabilities, columns=['Negative_Class', 'Positive_Class'], index=X_test.index)

# Convert DataFrame to CSV
probabilities_csv = probabilities_df.to_csv(index=True)

date_str = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
filename = "svm_probabilities_new.csv"

# Save to Object Store
if qb.ObjectStore.Save(filename, probabilities_csv):
    print("Saved to object store successfully.")
else:
    print("Failed to save to object store.")

# %%
val_report = classification_report(y_val, y_val_pred, output_dict=True)
test_report = classification_report(y_test, y_test_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
classes = ['0', '1', 'weighted avg']
val_df = pd.DataFrame(val_report).T.loc[classes, metrics]
test_df = pd.DataFrame(test_report).T.loc[classes, metrics]

# metrics for comparison
x = np.arange(len(classes))  # Class labels (0 and 1)

# Plot settings
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Comparison of Precision, Recall, and F1-score (Validation vs Test Set)', fontsize=16)

# Precision comparison
axes[0].bar(x - 0.2, val_df['precision'], width=0.4, label='Validation', color='blue')
axes[0].bar(x + 0.2, test_df['precision'], width=0.4, label='Test', color='orange')
axes[0].set_title('Precision')
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].legend()

# Recall comparison
axes[1].bar(x - 0.2, val_df['recall'], width=0.4, label='Validation', color='blue')
axes[1].bar(x + 0.2, test_df['recall'], width=0.4, label='Test', color='orange')
axes[1].set_title('Recall')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)

# F1-score comparison
axes[2].bar(x - 0.2, val_df['f1-score'], width=0.4, label='Validation', color='blue')
axes[2].bar(x + 0.2, test_df['f1-score'], width=0.4, label='Test', color='orange')
axes[2].set_title('F1-Score')
axes[2].set_xticks(x)
axes[2].set_xticklabels(classes)
for ax in axes:
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()


