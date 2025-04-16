import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


# Load CSV file
def load_data(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["Date"])  # Parse date column
    df.set_index("Date", inplace=True)  # Set date as index
    return df


# Preprocess data
def preprocess_data(df, sequence_length=10):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[["kWh"]])  # Scale only kWh column
    sequences = []
    for i in range(len(data_scaled) - sequence_length):
        sequences.append(data_scaled[i : i + sequence_length])
    return np.array(sequences), scaler


# Create LSTM autoencoder model
def build_lstm_autoencoder(sequence_length, feature_dim):
    model = keras.Sequential(
        [
            LSTM(
                32,
                activation="relu",
                input_shape=(sequence_length, feature_dim),
                return_sequences=True,
            ),
            LSTM(16, activation="relu", return_sequences=False),
            RepeatVector(sequence_length),
            LSTM(16, activation="relu", return_sequences=True),
            LSTM(32, activation="relu", return_sequences=True),
            TimeDistributed(Dense(feature_dim)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# Train the model
def train_model(autoencoder, x_train, x_val, epochs=20, batch_size=32):
    autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, x_val),
        verbose=1,
    )
    return autoencoder


# Save the model
def save_model(
    autoencoder, scaler, model_path="autoencoder_model.keras", scaler_path="scaler.pkl"
):
    autoencoder.save(model_path)
    import joblib

    joblib.dump(scaler, scaler_path)


# Main execution
if __name__ == "__main__":
    csv_file = "data/solar_output.csv"  # Change this to your CSV file
    df = load_data(csv_file)
    sequence_length = 10  # Use sequences of 10 time steps
    data_sequences, scaler = preprocess_data(df, sequence_length)

    # Split data
    x_train, x_val = train_test_split(data_sequences, test_size=0.2, random_state=42)

    # Build and train the LSTM-based model
    autoencoder = build_lstm_autoencoder(sequence_length, x_train.shape[2])
    autoencoder = train_model(autoencoder, x_train, x_val)

    # Save the model and scaler
    save_model(autoencoder, scaler)
    print("LSTM-based model and scaler saved successfully.")
