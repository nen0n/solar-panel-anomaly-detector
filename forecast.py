import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model

# 1. Load the solar output data
data = pd.read_csv("data/solar_output.csv")

# Convert 'Date' to datetime and set it as the index
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Select only the 'kWh' column for forecasting
data = data[["kWh"]]

# 2. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 3. Prepare sequences for the autoencoder model (using a sliding window approach)
sequence_length = 10
X = []
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i - sequence_length : i])

X = np.array(X)

# 4. Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

# 5. Load the pre-trained autoencoder model
autoencoder = load_model("autoencoder_model.keras")

# 6. Make predictions using the trained model
predictions = autoencoder.predict(X_test)

# 7. Invert scaling to get the actual values
predicted_values = scaler.inverse_transform(predictions.reshape(-1, 1))
original_test_values = scaler.inverse_transform(X_test.reshape(-1, 1))

# 9. Create subplots for actual, predicted
plt.figure(figsize=(18, 12))

# Subplot 1: Actual Solar Output
plt.subplot(2, 1, 1)
plt.plot(original_test_values, label="Actual Solar Output", color="blue")
plt.title("Actual Solar Output")
plt.xlabel("Timestamps")
plt.ylabel("Solar Output (kWh)")
plt.legend()

# Subplot 2: Predicted Solar Output
plt.subplot(2, 1, 2)
plt.plot(predicted_values, label="Predicted Solar Output", color="green")
plt.title("Predicted Solar Output")
plt.xlabel("Timestamps")
plt.ylabel("Solar Output (kWh)")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
