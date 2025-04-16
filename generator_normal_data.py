# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


# Function to simulate solar panel output with added noise
def solar_panel_output_with_noise(
    time,
    start_month=1,
    num_days=20,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
):
    # Dictionary of average daylight hours for each month (simplified)
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

    # Create a time array for the simulation duration
    total_time = np.linspace(0, num_days * 24, num_days * 1000)
    output = np.zeros_like(total_time)

    # Simulate output based on sinusoidal function and daylight scaling
    for i, t in enumerate(total_time):
        month = start_month + (int(t // 24) % num_days)
        scale_factor = daylight_hours.get(month, 12) / 12
        output[i] = (
            scale_factor * max_output * np.sin((2 * np.pi * (t - phase_shift)) / period)
        )

    # Add Gaussian noise to simulate variability
    noise = np.random.normal(0, noise_level, len(output))
    output += noise

    # Ensure no negative output (as solar panels can't go below zero)
    output = np.maximum(output, 0)

    return total_time, output


# Function to generate and save solar output data to CSV
def generate_and_save_solar_output(
    start_date="2025-03-01",
    num_months=20,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
    filename="solar_output.csv",
):
    # Dictionary of average daylight hours for each month (simplified)
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

    # Create a time array (in hours) for simulation
    total_time = np.linspace(0, num_months * 24, num_months * 1000)
    output = np.zeros_like(total_time)

    # Simulate output with scaled sine wave and daylight variation
    for i, t in enumerate(total_time):
        month = 1 + (int(t // 24) % num_months)
        scale_factor = daylight_hours.get(month, 12) / 12
        output[i] = (
            scale_factor * max_output * np.sin((2 * np.pi * (t - phase_shift)) / period)
        )

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(output))
    output += noise

    # Prevent negative values
    output = np.maximum(output, 0)

    # Create datetime timestamps based on start date
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [base_date + timedelta(hours=t) for t in total_time]

    # Assume output is in kWh (adjust scaling if needed)
    kWh_output = output * 1

    # Create DataFrame and save to CSV
    df = pd.DataFrame({"Date": timestamps, "kWh": kWh_output})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Main entry point of the script
def main():
    # Generate and export simulated solar data to a file
    generate_and_save_solar_output(
        start_date="2025-03-01",
        num_months=20,
        filename="C:/Users/gzdes/PycharmProjects/Diplome/data/solar_output.csv",
    )

    # Generate one day of noisy solar output for visualization
    time, output = solar_panel_output_with_noise(time=np.linspace(0, 24, 1000))

    # Plot the output
    plt.plot(time, output, label="Solar Panel Output with Noise")
    plt.title("Simulated Solar Panel Output with Noise Over 20 Days")
    plt.xlabel("Time (hours)")
    plt.ylabel("Output (relative to max output)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Only run the main function if the script is executed directly
if __name__ == "__main__":
    main()
