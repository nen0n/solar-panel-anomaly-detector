import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def simulate_and_save_solar_output(
    start_date="2025-03-01",
    num_days=20,
    max_output=1,
    period=24,
    phase_shift=0,
    noise_level=0.05,
    filename="solar_output.csv",
    plot_day=True,
    plot_full=True
):
    # Dictionary of average daylight hours (extendable)
    daylight_hours = {
        1: 9, 2: 10, 3: 12, 4: 14, 5: 15, 6: 15, 7: 14, 8: 13, 9: 14, 10: 13,
        11: 12, 12: 12, 13: 13, 14: 14, 15: 13, 16: 11, 17: 9, 18: 10, 19: 11, 20: 13,
    }

    # Time array in hours
    total_time = np.linspace(0, num_days * 24, num_days * 1000)
    output = np.zeros_like(total_time)

    # Simulate solar output
    for i, t in enumerate(total_time):
        month = 1 + (int(t // 24) % num_days)
        scale_factor = daylight_hours.get(month, 12) / 12
        output[i] = scale_factor * max_output * np.sin((2 * np.pi * (t - phase_shift)) / period)

    # Add noise and clamp negatives
    output += np.random.normal(0, noise_level, len(output))
    output = np.maximum(output, 0)

    # Generate timestamps
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [base_date + timedelta(hours=t) for t in total_time]

    # Save data
    df = pd.DataFrame({"Date": timestamps, "kWh": output})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

    # Plot one day (optional)
    if plot_day:
        day_time = np.linspace(0, 24, 1000)
        scale_factor = daylight_hours.get(1, 12) / 12
        day_output = scale_factor * max_output * np.sin((2 * np.pi * (day_time - phase_shift)) / period)
        day_output += np.random.normal(0, noise_level, len(day_output))
        day_output = np.maximum(day_output, 0)

        plt.figure(figsize=(10, 4))
        plt.plot(day_time, day_output, label="One Day Output", color="orange")
        plt.title("Simulated Solar Output (1 Day with Noise)")
        plt.xlabel("Time (hours)")
        plt.ylabel("Output (relative to max)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot full dataset (optional)
    if plot_full:
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, output, label="Full Simulation Output", color="blue", linewidth=0.8)
        plt.title(f"Simulated Solar Output Over {num_days} Days")
        plt.xlabel("Date")
        plt.ylabel("kWh Output")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Run if executed directly
if __name__ == "__main__":
    simulate_and_save_solar_output(
        start_date="2025-03-01",
        num_days=20,
        filename="data/solar_output.csv"
    )
