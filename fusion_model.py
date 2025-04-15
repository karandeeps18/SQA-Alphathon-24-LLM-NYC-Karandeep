# %% [markdown]
# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
# <hr>

# %%
import pandas as pd
from io import StringIO
from QuantConnect import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


# Initialize QuantBook
qb = QuantBook()

# stored SVM probabilities
svm_probabilities_csv = qb.ObjectStore.Read("svm_probabilities_new.csv")
if svm_probabilities_csv:
    svm_probabilities = pd.read_csv(StringIO(svm_probabilities_csv))

# CNN probabilities
cnn_probabilities_csv = qb.ObjectStore.Read("cnn_predictions_with_date.csv")
if cnn_probabilities_csv:
    cnn_probabilities = pd.read_csv(StringIO(cnn_probabilities_csv))


# %%
svm_probabilities['time'] = pd.to_datetime(svm_probabilities['time']).dt.date
cnn_probabilities['DateTime'] = pd.to_datetime(cnn_probabilities['DateTime']).dt.date

# Merge the data frames on the date column
merged_probabilities = pd.merge(svm_probabilities, cnn_probabilities, left_on='time', right_on='DateTime', how='inner')

merged_probabilities


# %%
# Initialize QuantBook
qb = QuantBook()

# Add TSLA equity
tsla_symbol = qb.AddEquity("TSLA").Symbol
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)
tsla_history = qb.History(tsla_symbol, start_date, end_date, Resolution.Daily)
tsla_history['pct_change'] = tsla_history['close'].pct_change() #pct change

# reset index
if 'level_0' in tsla_history.columns:
    tsla_history.drop(columns=['level_0'], inplace=True)
if 'index' in tsla_history.columns:
    tsla_history.drop(columns=['index'], inplace=True)

tsla_history.reset_index(inplace=True)
tsla_history['time'] = pd.to_datetime(tsla_history['time']).dt.date
merged_probabilities['time'] = pd.to_datetime(merged_probabilities['time']).dt.date
merged_probabilities.set_index('time', inplace=True)
combined_data = merged_probabilities.join(tsla_history.set_index('time')[['pct_change']], how='inner')

combined_data


# %%
combined_data['Target'] = (combined_data['pct_change'] > 0.04).astype(int)
features = combined_data[['Positive_Class', 'Predictions']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, combined_data['Target'], test_size=0.2, random_state=42)

# SVM with a radial basis function (RBF) kernel
classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')

# Training the classifier
classifier.fit(X_train, y_train)

# predictions
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)[:, 1]  # probabilities for the positive class

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# %%
# Save to ObjectStore
dates = pd.date_range(start='2019-01-01', periods=len(predictions), freq='D')
predictions_df = pd.DataFrame({
    'DateTime': dates,
    'Predictions': predictions
})

predictions_csv = predictions_df.to_csv(index=False)
predictions_bytes = predictions_csv.encode()
if qb.ObjectStore.SaveBytes("fusion_predictions.csv", predictions_bytes):
    print("Saved successfully.")
else:
    print("Failed to save.")


