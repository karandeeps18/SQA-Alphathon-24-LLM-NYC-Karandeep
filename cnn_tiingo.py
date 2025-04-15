# %% [markdown]
# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
# <hr>

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from datetime import datetime
import re
from QuantConnect import *
from QuantConnect.Data.Custom.Tiingo import *
from datetime import datetime
from QuantConnect.DataSource import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data.Custom.Tiingo import *
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json


# %%
# Initialize QuantBook
qb = QuantBook()

# List of tickers
tickers = ["TSLA"]

tiingo_news_objects = {}
for ticker in tickers:
    equity = qb.AddEquity(ticker) 
    tiingo_news_objects[ticker] = qb.AddData(TiingoNews, equity.Symbol) 

start_date = datetime(2019, 6, 1)
end_date = datetime(2019, 12, 31)


for ticker in tickers:
    symbol = tiingo_news_objects[ticker].Symbol
    news_history = qb.History(symbol, start_date, end_date, Resolution.Daily)
    print(f"News data for {ticker}:")
    if not news_history.empty:
        print(news_history.head())
    else:
        print("No news data available.")

# %%
news_history['crawldate'] = pd.to_datetime(news_history['crawldate'], utc=True)
news_history

# %% [markdown]
# ### Data Cleaning and setup 

# %%
def clean(text):
    text = text.lower()  # to lower case
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Removing non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip() 
    return text 

news_history['description_clean'] = news_history['description'].apply(clean)
news_history['title_clean'] = news_history['title'].apply(clean)

# num entries 
num_entries = len(news_history)
print(f"Total number of entries: {num_entries}")
print(f"Description Cleaned: {news_history['description_clean']}")

# %% [markdown]
# ## Generating Embeddings 

# %%
#difing focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

# %%
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = TFBertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# %%
def predict_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(encoded_input)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    return 'positive' if predictions[0, 1] > predictions[0, 0] else 'negative'
# setiment analysis 
news_history['sentiment_label'] = news_history['description_clean'].apply(predict_sentiment)

# %%
labels = news_history['sentiment_label'].map({'positive': 1, 'negative': 0}).dropna()
labels

# %%
finbert = TFBertModel.from_pretrained('yiyanghkust/finbert-tone', from_pt=True)

# %%
# Function to generate embeddings
def generate_embeddings(text_list, batch_size=16):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]  
        if isinstance(batch_texts, list) and all(isinstance(item, str) for item in batch_texts):
            encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='tf')
            outputs = finbert(encoded_inputs)
            # the pooled output representations at CLS token
            batch_embeddings = outputs.pooler_output.numpy()
            embeddings.append(batch_embeddings)
        else:
            raise ValueError("Batch texts must be a list of strings")
    embeddings = np.vstack(embeddings)
    embeddings = embeddings.reshape(-1, 1, 768) # addings dummy sequential dimension for CNN 
    return np.vstack(embeddings)
embeddings = generate_embeddings(news_history['description_clean'].tolist())
print(f"Generated embeddings shape: {embeddings.shape}")

# %%
embeddings_reshaped = embeddings.reshape(-1, 1, 768)
print(f"Generated embeddings_reshaped shape: {embeddings_reshaped.shape}")

# %% [markdown]
# ## Defining CNN Architecture

# %%
def parallel_cnn(input_shape):
    # input layer
    inputs = Input(shape=input_shape)


    # Parallel Convolution Layers with different Kernel Size 
    conv_3 = Conv1D(filters=128, kernel_size=1, activation='relu', strides=1, padding='valid')(inputs)
    conv_4 = Conv1D(filters=128, kernel_size=1, activation='relu', strides=1, padding='valid')(inputs)
    conv_5 = Conv1D(filters=128, kernel_size=1, activation='relu', strides=1, padding='valid')(inputs)

    pool_size_3 = conv_3.shape[1]
    pool_size_4 = conv_4.shape[1]
    pool_size_5 = conv_5.shape[1]


    #MaxPooling for each Convolution output 
    maxpool_3 = MaxPooling1D(pool_size=pool_size_3)(conv_3)   
    maxpool_4 = MaxPooling1D(pool_size=pool_size_4)(conv_4)   
    maxpool_5 = MaxPooling1D(pool_size=pool_size_5)(conv_5)   

    # Concarenating the pooled features 

    concatenated = Concatenate()(([maxpool_3, maxpool_4, maxpool_5]))
    flat = Flatten()(concatenated)

    #Fully connected layers
    fc1 = Dense(120, activation='relu')(flat)
    dropout = Dropout(0.5)(fc1)
    fc2 = Dense(84, activation='relu')(dropout)

    # Output Layer 
    output = Dense(1, activation='softmax')(fc2)
    model= Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model 

# model initialtion 
model = parallel_cnn(input_shape=(1, 768))
model.summary()

# %%
model = parallel_cnn(input_shape=(1, 768))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings_reshaped, labels, test_size=0.2, random_state=42)

# Train 
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# %%
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# %%
# Predict using the model
predictions = model.predict(X_test)
rounded_predictions = np.round(predictions)

# %%
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='darkorange', label=f'PR curve (area = {pr_auc:.2f})')
plt.fill_between(recall, precision, alpha=0.2, color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="best")
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, rounded_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# %%
dates = pd.date_range(start='2019-06-01', periods=len(predictions), freq='D')
predictions_df = pd.DataFrame({
    'DateTime': dates,
    'Predictions': predictions.flatten()
})

predictions_csv = predictions_df.to_csv(index=False)

# Encode to bytes for saving
predictions_bytes = predictions_csv.encode()

# Save to Object Store
if qb.ObjectStore.SaveBytes("cnn_predictions_with_date.csv", predictions_bytes):
    print("Saved successfully.")
else:
    print("Failed to save.")


