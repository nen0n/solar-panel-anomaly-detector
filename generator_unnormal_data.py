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

    Args:
    - start_date (str): Start date of the simulation (default is "2025-03-01").
    - num_months (int): The number of months to simulate (default is 12).
    - max_output (float): Maximum output of the solar panel (default is 1).
    - period (float): Time period of the cycle (default is 24 hours).
    - phase_shift (float): Phase shift to adjust the timing of the peak output (default is 0).
    - noise_level (float): Standard deviation of the noise to be added (default is 0.05).
    - anomaly_rate (float): Probability of an anomaly occurring at each time point (default is 0.01).
    - num_datasets (int): Number of datasets to generate (default is 3).
    - base_filename (str): Base filename for saving the datasets (default is "solar_output_with_varied_anomalies").
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

    # Generate the continuous time array for the entire period (in hours)
    total_time = np.linspace(
        0, num_months * 24, num_months * 1000
    )  # 1000 time points per month

    for dataset_num in range(1, num_datasets + 1):
        output = np.zeros_like(total_time)

        # Calculate the solar panel output for each time point
        for i, t in enumerate(total_time):
            month = 1 + (int(t // 24) % num_months)
            scale_factor = daylight_hours.get(month, 12) / 12
            output[i] = (
                scale_factor
                * max_output
                * np.sin((2 * np.pi * (t - phase_shift)) / period)
            )

        # Add random noise
        noise = np.random.normal(0, noise_level, len(output))  # Gaussian noise
        output += noise

        # Dataset 1: Normal dataset with anomalies
        if dataset_num == 1:
            anomaly_indices = (
                np.random.rand(len(output)) < anomaly_rate
            )  # Randomly select anomaly points
            output[anomaly_indices] = np.random.uniform(
                0, max_output * 2, np.sum(anomaly_indices)
            )  # Add anomalies
        # Dataset 2: Increased output (scaled by 2)
        elif dataset_num == 2:
            output *= 2  # Scale the output by a factor of 2
        # Dataset 3: Output with sections of 0 or 1
        elif dataset_num == 3:
            # Create several continuous sections where output is 0 or 1
            segment_length = len(output) // 10  # Divide data into 10 segments
            for i in range(0, len(output), segment_length):
                if np.random.rand() < 0.5:  # 50% chance to set the segment to 0 or 1
                    output[i : i + segment_length] = np.random.choice(
                        [0, 1]
                    )  # Set the whole segment to 0 or 1

        # Ensure output is non-negative
        output = np.maximum(output, 0)

        # Generate corresponding timestamps for each time point
        base_date = datetime.strptime(start_date, "%Y-%m-%d")
        timestamps = [base_date + timedelta(hours=t) for t in total_time]

        # Convert output to kWh
        kwh_output = (
            output * 1
        )  # You can adjust this if necessary based on actual panel capacity

        # Create a DataFrame
        data = {"Date": timestamps, "kWh": kwh_output}

        # Save to CSV
        filename = f"{base_filename}_{dataset_num}.csv"
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        # Plot the output and anomalies
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, kwh_output, label="Solar Output", color="b")

        # if dataset_num == 1:
        #     # Anomalies for dataset 1
        #     anomaly_indices = np.random.rand(len(output)) < anomaly_rate
        #     plt.scatter(np.array(timestamps)[anomaly_indices], kWh_output[anomaly_indices], color='r',
        #                 label="Anomalies", zorder=5)

        plt.title(f"Solar Panel Output with Varied Anomalies - Dataset {dataset_num}")
        plt.xlabel("Date")
        plt.ylabel("kWh")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Show only the first 3 datasets
        if dataset_num <= 3:
            plt.show()


# Example usage
generate_solar_with_varied_anomalies_and_plot(
    start_date="2025-03-01",
    num_months=20,
    num_datasets=3,
    base_filename="data\solar_output_with_varied_anomalies",
)
