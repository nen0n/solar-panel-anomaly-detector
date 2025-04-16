import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare_data(file_path):
    """Load CSV and convert 'Date' column to datetime."""
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def plot_data(normal_data, anomalous_data_list):
    """Plot normal and anomalous data using matplotlib."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    titles = ["Normal Data", "Anomalous Data 1", "Anomalous Data 2", "Anomalous Data 3"]
    colors = ["blue", "red", "orange", "green"]
    datasets = [normal_data] + anomalous_data_list

    for ax, data, title, color in zip(axs.flat, datasets, titles, colors):
        ax.plot(data["Date"], data["kWh"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("kWh")

    plt.show()


def main():
    # Load datasets
    normal_data = load_and_prepare_data("data/solar_output.csv")
    anomalous_data_1 = load_and_prepare_data(
        "data/solar_output_with_varied_anomalies_1.csv"
    )
    anomalous_data_2 = load_and_prepare_data(
        "data/solar_output_with_varied_anomalies_2.csv"
    )
    anomalous_data_3 = load_and_prepare_data(
        "data/solar_output_with_varied_anomalies_3.csv"
    )

    # Preview datasets
    print("Normal Data Preview:\n", normal_data.head())
    print("Anomalous Data 1 Preview:\n", anomalous_data_1.head())
    print("Anomalous Data 2 Preview:\n", anomalous_data_2.head())
    print("Anomalous Data 3 Preview:\n", anomalous_data_3.head())

    # Plot data
    plot_data(normal_data, [anomalous_data_1, anomalous_data_2, anomalous_data_3])


if __name__ == "__main__":
    main()
