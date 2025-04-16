import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model


def load_and_prepare_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data = data[["kWh"]]
    return data


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


def create_sequences(data_scaled, sequence_length):
    x = []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i - sequence_length : i])
    return np.array(x)


def plot_results(original, predicted):
    plt.figure(figsize=(18, 12))

    # Actual values
    plt.subplot(2, 1, 1)
    plt.plot(original, label="Actual Solar Output", color="blue")
    plt.title("Actual Solar Output")
    plt.xlabel("Timestamps")
    plt.ylabel("Solar Output (kWh)")
    plt.legend()

    # Predicted values
    plt.subplot(2, 1, 2)
    plt.plot(predicted, label="Predicted Solar Output", color="green")
    plt.title("Predicted Solar Output")
    plt.xlabel("Timestamps")
    plt.ylabel("Solar Output (kWh)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 1. Load and prepare data
    data = load_and_prepare_data("data/solar_output.csv")

    # 2. Normalize data
    data_scaled, scaler = normalize_data(data)

    # 3. Create sequences
    sequence_length = 10
    x_output = create_sequences(data_scaled, sequence_length)

    # 4. Train-test split
    x_train, x_test = train_test_split(x_output, test_size=0.2, shuffle=False)

    # 5. Load pre-trained model
    autoencoder = load_model("autoencoder_model.keras")

    # 6. Make predictions
    predictions = autoencoder.predict(x_test)

    # 7. Invert scaling
    predicted_values = scaler.inverse_transform(predictions.reshape(-1, 1))
    original_test_values = scaler.inverse_transform(x_test.reshape(-1, 1))

    # 8. Plot results
    plot_results(original_test_values, predicted_values)


if __name__ == "__main__":
    main()
