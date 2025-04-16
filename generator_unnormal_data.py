import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def generate_solar_with_varied_anomalies_and_plot(
    start_date="2025-03-01",
    num_months=12,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
    anomaly_rate=0.01,
    num_datasets=3,
    base_filename="solar_output_with_varied_anomalies",
):
    """
    Generates multiple solar panel output datasets with varied anomalies and saves them into separate CSV files.
    Visualizes the anomalies in the first three datasets.
    """
    daylight_hours = {
        1: 9,
        2: 10,
        3: 12,
        4: 14,
        5: 15,
        6: 15,
        7: 14,
        8: 13,
        9: 12,
        10: 11,
        11: 9,
        12: 9,
    }

    total_time = np.linspace(0, num_months * 24, num_months * 1000)

    for dataset_num in range(1, num_datasets + 1):
        output = np.zeros_like(total_time)

        for i, t in enumerate(total_time):
            month = 1 + (int(t // 24) % num_months)
            scale_factor = daylight_hours.get(month, 12) / 12
            output[i] = (
                scale_factor
                * max_output
                * np.sin((2 * np.pi * (t - phase_shift)) / period)
            )

        noise = np.random.normal(0, noise_level, len(output))
        output += noise

        if dataset_num == 1:
            anomaly_indices = np.random.rand(len(output)) < anomaly_rate
            output[anomaly_indices] = np.random.uniform(
                0, max_output * 2, np.sum(anomaly_indices)
            )
        elif dataset_num == 2:
            output *= 2
        elif dataset_num == 3:
            segment_length = len(output) // 10
            for i in range(0, len(output), segment_length):
                if np.random.rand() < 0.5:
                    output[i : i + segment_length] = np.random.choice([0, 1])

        output = np.maximum(output, 0)

        base_date = datetime.strptime(start_date, "%Y-%m-%d")
        timestamps = [base_date + timedelta(hours=t) for t in total_time]
        kwh_output = output

        data = {"Date": timestamps, "kWh": kwh_output}
        filename = f"{base_filename}_{dataset_num}.csv"
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, kwh_output, label="Solar Output", color="b")
        plt.title(f"Solar Panel Output with Varied Anomalies - Dataset {dataset_num}")
        plt.xlabel("Date")
        plt.ylabel("kWh")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if dataset_num <= 3:
            plt.show()


if __name__ == "__main__":
    generate_solar_with_varied_anomalies_and_plot(
        start_date="2025-03-01",
        num_months=20,
        num_datasets=3,
        base_filename="data/solar_output_with_varied_anomalies",
    )
