import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


def solar_panel_output_with_noise(
    time,
    start_month=1,
    num_days=20,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
):
    """
    Simulates the output of a solar panel over several months with added random noise.

    Args:
    - time (array): Array of time values (e.g., in hours, for a full day 0-24).
    - start_month (int): The month to start the simulation (1 = January).
    - num_months (int): The number of months to simulate.
    - max_output (float): Maximum output of the solar panel (default is 1).
    - period (float): Time period of the cycle (default is 24 hours).
    - phase_shift (float): Phase shift to adjust the timing of the peak output (default is 0).
    - noise_level (float): Standard deviation of the noise to be added (default is 0.05).

    Returns:
    - output (array): Solar panel output over the specified period, with added noise.
    """
    # Approximate daylight hours per month (simplified model)
    daylight_hours = {
        1: 9,
        2: 10,
        3: 12,
        4: 14,
        5: 15,
        6: 15,
        7: 14,
        8: 13,
        9: 14,
        10: 13,
        11: 12,
        12: 12,
        13: 13,
        14: 14,
        15: 13,
        16: 11,
        17: 9,
        18: 10,
        19: 11,
        20: 13,
    }

    # Generate the continuous time array for the entire period (in hours)
    total_time = np.linspace(
        0, num_days * 24, num_days * 1000
    )  # 1000 time points per month

    # Initialize an empty array to store the output values
    output = np.zeros_like(total_time)

    # Calculate the solar panel output for each time point
    for i, t in enumerate(total_time):
        month = start_month + (int(t // 24) % num_days)
        scale_factor = daylight_hours.get(month, 12) / 12
        output[i] = (
            scale_factor * max_output * np.sin((2 * np.pi * (t - phase_shift)) / period)
        )

    # Add random noise to simulate variability
    noise = np.random.normal(0, noise_level, len(output))  # Gaussian noise
    output += noise

    # Ensure output is non-negative (solar panels can't produce negative output)
    output = np.maximum(output, 0)

    return total_time, output


def generate_and_save_solar_output(
    start_date="2025-03-01",
    num_months=20,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
    filename="solar_output.csv",
):
    """
    Generates solar panel output with noise and saves the data into a CSV file.

    Args:
    - start_date (str): Start date of the simulation (default is "2025-03-01").
    - num_months (int): The number of months to simulate (default is 12).
    - max_output (float): Maximum output of the solar panel (default is 1).
    - period (float): Time period of the cycle (default is 24 hours).
    - phase_shift (float): Phase shift to adjust the timing of the peak output (default is 0).
    - noise_level (float): Standard deviation of the noise to be added (default is 0.05).
    - filename (str): Name of the CSV file to save (default is "solar_output.csv").
    """

    # Approximate daylight hours per month (simplified model)
    daylight_hours = {
        1: 9,
        2: 10,
        3: 12,
        4: 14,
        5: 15,
        6: 15,
        7: 14,
        8: 13,
        9: 14,
        10: 13,
        11: 12,
        12: 12,
        13: 13,
        14: 14,
        15: 13,
        16: 11,
        17: 9,
        18: 10,
        19: 11,
        20: 13,
    }

    # Generate the continuous time array for the entire period (in hours)
    total_time = np.linspace(
        0, num_months * 24, num_months * 1000
    )  # 1000 time points per month

    # Initialize an empty array to store the output values
    output = np.zeros_like(total_time)

    # Calculate the solar panel output for each time point
    for i, t in enumerate(total_time):
        month = 1 + (int(t // 24) % num_months)
        scale_factor = daylight_hours.get(month, 12) / 12
        output[i] = (
            scale_factor * max_output * np.sin((2 * np.pi * (t - phase_shift)) / period)
        )

    # Add random noise to simulate variability
    noise = np.random.normal(0, noise_level, len(output))  # Gaussian noise
    output += noise

    # Ensure output is non-negative
    output = np.maximum(output, 0)

    # Generate corresponding timestamps for each time point
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [base_date + timedelta(hours=t) for t in total_time]

    # Convert output to kWh (assuming max_output = 1, so output is in relative kWh)
    kWh_output = (
        output * 1
    )  # You can adjust this if necessary based on actual panel capacity

    # Create a DataFrame
    data = {"Date": timestamps, "kWh": kWh_output}

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Example usage
generate_and_save_solar_output(
    start_date="2025-03-01",
    num_months=20,
    filename="C:/Users/gzdes/PycharmProjects/Diplome/data/solar_output.csv",
)

# Example usage
time, output = solar_panel_output_with_noise(time=np.linspace(0, 24, 1000))

# Plot the continuous output with noise over 12 months
plt.plot(time, output, label="Solar Panel Output with Noise")
plt.title("Simulated Solar Panel Output with Noise Over 20 Days")
plt.xlabel("Time (hours)")
plt.ylabel("Output (relative to max output)")
plt.grid(True)
plt.show()
