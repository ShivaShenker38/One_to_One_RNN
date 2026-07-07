# ==========================================
# Import Libraries
# ==========================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

# ==========================================
# Create Folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("scaler", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

print("Loading Dataset...")

df = pd.read_csv("NFLX.csv")

print(df.head())

print(df.info())

# ==========================================
# Select Close Price
# ==========================================

data = df[['Close']]

print(data.head())

# ==========================================
# Normalize Data
# ==========================================

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

# Save Scaler

pickle.dump(
    scaler,
    open(
        "scaler/scaler.pkl",
        "wb"
    )
)

print("Scaler Saved")

# ==========================================
# Create Sequences
# ==========================================

time_step = 30

X = []
Y = []

for i in range(time_step, len(scaled_data)):

    X.append(
        scaled_data[i-time_step:i,0]
    )

    Y.append(
        scaled_data[i,0]
    )

X = np.array(X)

Y = np.array(Y)

print("X Shape :", X.shape)

print("Y Shape :", Y.shape)

# ==========================================
# Reshape for RNN
# ==========================================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print("Reshaped X :", X.shape)

# ==========================================
# Train Test Split
# ==========================================

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]

print()

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))
# ==========================================
# Build RNN Model
# ==========================================

model = Sequential()

model.add(
    SimpleRNN(
        units=50,
        activation="tanh",
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(Dense(25))

model.add(Dense(1))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

print("\nModel Summary\n")

model.summary()

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)

# ==========================================
# Predict
# ==========================================

predictions = model.predict(X_test)

# Convert back to original values

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(
    Y_test.reshape(-1,1)
)

# ==========================================
# Evaluate
# ==========================================

loss, mae = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)

print("\nTest Loss :", loss)
print("Mean Absolute Error :", mae)

# ==========================================
# Save Model
# ==========================================

model.save("models/netflix_rnn.keras")

print("Model Saved Successfully")

# ==========================================
# Plot Loss
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ==========================================
# Plot Actual vs Predicted
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(actual, label="Actual Close Price")

plt.plot(predictions, label="Predicted Close Price")

plt.title("Netflix Stock Price Prediction")

plt.xlabel("Days")

plt.ylabel("Close Price")

plt.legend()

plt.show()

print("\nTraining Completed Successfully")