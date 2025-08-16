import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Load data

df = pd.read_csv('Practical Development Programming/videostats.csv', nrows=2000)   
print("Loaded shape:", df.shape)
print(df.columns.tolist())
print("Dataset Shape:", df.shape)
print(df.head())
print("Info(): ", df.info())
print("datatype(): ", df.dtypes)
print("describe(): ", df.describe())

# Choose series to model
# Select ytvideoid with maximum records in the first 2000 rows 
top_video = df['ytvideoid'].value_counts().idxmax()
video_df = df[df['ytvideoid'] == top_video].sort_index().reset_index(drop=True)
print(f"Top ytvideoid: {top_video}  |  #records: {len(video_df)}")
print(video_df.head())

# Use 'views' column as the target series.
# If that specific video has very few records (< 60), fallback to using the overall 'views' series from first 2000 rows.
series = video_df['views'].astype(float).values.reshape(-1, 1)
if len(series) < 60:
    print("Selected video has <60 points. Falling back to entire 2000-row 'views' series.")
    series = df['views'].astype(float).values.reshape(-1, 1)

print("Final series length:", len(series))

# Scale values
scaler = MinMaxScaler(feature_range=(0,1))
series_scaled = scaler.fit_transform(series)

# Create sliding window sequences
def create_sequences(arr, window_size=30):
    X, y = [], []
    for i in range(len(arr) - window_size):
        X.append(arr[i:i+window_size])
        y.append(arr[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

window_size = 30  
X, y = create_sequences(series_scaled, window_size)
print("Samples:", X.shape)

# Ensure shape for models: samples, timesteps, features
# Our features = 1 views
# X already in that shape: (n_samples, window_size, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# Models definitions

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_dense_baseline(input_shape):
    # flatten the window and use Dense layers as a simple baseline
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# 8. Training
input_shape = (X_train.shape[1], X_train.shape[2])
early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

epochs = 40
batch_size = 16

# LSTM
lstm = build_lstm(input_shape)
history_lstm = lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early], verbose=1)

# GRU
gru = build_gru(input_shape)
history_gru = gru.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_test, y_test), callbacks=[early], verbose=1)

# Dense baseline
dense = build_dense_baseline(input_shape)
history_dense = dense.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_data=(X_test, y_test), callbacks=[early], verbose=1)


# 9. Evaluate (MAE, RMSE) and Inverse transform predictions
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def evaluate_and_inverse(model, X_test, y_test, scaler):
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    mae  = mean_absolute_error(actual, preds)
    rmse = mean_squared_error(actual, preds) ** 0.5
    return mae, rmse, preds, actual

# Evaluate all three models
l_mae, l_rmse, l_preds, l_actual = evaluate_and_inverse(lstm,  X_test, y_test, scaler)
g_mae, g_rmse, g_preds, g_actual = evaluate_and_inverse(gru,   X_test, y_test, scaler)
d_mae, d_rmse, d_preds, d_actual = evaluate_and_inverse(dense, X_test, y_test, scaler)

# Create results DataFrame
results = pd.DataFrame({
    'Model':        ['LSTM', 'GRU', 'Dense-Baseline'],
    'MAE':          [l_mae,  g_mae,  d_mae],
    'RMSE':         [l_rmse, g_rmse, d_rmse],
    'Test Samples': [len(l_actual), len(g_actual), len(d_actual)]
})

print(results)

# Plots
# Training/Validation loss plot
plt.figure(figsize=(10,5))
plt.plot(history_lstm.history['loss'], label='LSTM train loss')
plt.plot(history_lstm.history['val_loss'], label='LSTM val loss')
plt.plot(history_gru.history['loss'], label='GRU train loss')
plt.plot(history_gru.history['val_loss'], label='GRU val loss')
plt.plot(history_dense.history['loss'], label='Dense train loss')
plt.plot(history_dense.history['val_loss'], label='Dense val loss')
plt.legend()
plt.title('Train/Val Loss (MSE) for all models')
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.show()

# Predictions vs Actual for each model (plot first N points)
N = min(200, len(l_actual))  

plt.figure(figsize=(10,4))
plt.plot(l_actual[:N], label='Actual')
plt.plot(l_preds[:N], label='LSTM Pred')
plt.legend()
plt.title('LSTM: Actual vs Predicted (first {} points)'.format(N))
plt.show()
