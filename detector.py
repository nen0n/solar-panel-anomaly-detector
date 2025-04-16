import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def detect_anomalies(data, threshold=0.4, window_size=10):
    # Calculate rolling mean (or median) as expected value
    rolling_mean = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    # Calculate the absolute difference between data points and the rolling mean
    deviation = np.abs(data - rolling_mean)
    # Flag points where deviation exceeds the threshold
    anomalies = deviation > threshold * rolling_mean
    return anomalies


def process_file(file, model, threshold, idx):
    # Read the CSV data
    data = pd.read_csv(file)
    # Drop non-numeric columns (e.g., timestamps or any categorical columns)
    data_numeric = data.select_dtypes(include=[np.number])

    # Preprocess data (scale it)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    # Predict the data with the autoencoder model
    predictions = model.predict(data_scaled)

    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)

    reconstruction_error = np.mean(np.abs(data_scaled - predictions), axis=1)
    local_anomalies = detect_anomalies(
        data_scaled.flatten(), threshold=threshold, window_size=10
    )

    plt.figure(figsize=(15, 10))

    # Plot 1: Reconstruction Error
    plt.subplot(2, 2, 1)
    plt.plot(reconstruction_error, label="Reconstruction Error")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Anomaly Threshold")
    plt.title(f"Reconstruction Error and Anomaly Detection Threshold (File {idx + 1})")
    plt.legend()

    # Plot 2: Data with Anomalies (using rolling mean)
    plt.subplot(2, 2, 2)
    plt.plot(data_scaled, label="Data")
    plt.scatter(
        np.where(local_anomalies)[0],
        data_scaled[local_anomalies],
        color="red",
        label="Anomalies",
    )
    plt.title(f"Data with Anomalies (File {idx + 1})")
    plt.legend()

    # Plot 3: Reconstruction Error Histogram
    plt.subplot(2, 2, 3)
    plt.hist(reconstruction_error, bins=50, color="b", alpha=0.7)
    plt.axvline(x=threshold, color="r", linestyle="--", label="Anomaly Threshold")
    plt.title(f"Reconstruction Error Distribution (File {idx + 1})")
    plt.legend()

    # Plot 4: Pie chart showing the amount of anomalies
    plt.subplot(2, 2, 4)
    anomaly_count = np.sum(local_anomalies)
    normal_count = len(local_anomalies) - anomaly_count

    labels = ["Anomalies", "Normal"]
    sizes = [anomaly_count, normal_count]
    colors = ["red", "blue"]
    explode = (0.1, 0)

    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    plt.title(f"Anomalies vs Normal Data (File {idx + 1})")

    plt.tight_layout()
    plt.show()

    print(
        f"Anomalies detected in File {idx + 1} at indices: {np.where(local_anomalies)[0]}"
    )


def main():
    model_path = "autoencoder_model.keras"

    # Define the reconstruction error threshold to detect anomalies
    threshold = (
        0.4  # Adjust this value as needed (percentage error for anomaly detection)
    )

    # List of CSV files to process
    csv_files = [
        "data/solar_output_with_varied_anomalies_1.csv",
        "data/solar_output_with_varied_anomalies_2.csv",
        "data/solar_output_with_varied_anomalies_3.csv",
    ]

    # Load the autoencoder model
    autoencoder = load_model(model_path)

    for idx, file in enumerate(csv_files):
        process_file(file, autoencoder, threshold, idx)


if __name__ == "__main__":
    main()
