import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
normal_data = pd.read_csv(
    "data/solar_output.csv"
)  # Replace with the actual normal data file path
anomalous_data_1 = pd.read_csv("data/solar_output_with_varied_anomalies_1.csv")
anomalous_data_2 = pd.read_csv("data/solar_output_with_varied_anomalies_2.csv")
anomalous_data_3 = pd.read_csv("data/solar_output_with_varied_anomalies_3.csv")

# Print out the first few rows to verify data
print("Normal Data Preview:\n", normal_data.head())
print("Anomalous Data 1 Preview:\n", anomalous_data_1.head())
print("Anomalous Data 2 Preview:\n", anomalous_data_2.head())
print("Anomalous Data 3 Preview:\n", anomalous_data_3.head())

# Convert the 'Date' column to datetime if it's not already
normal_data["Date"] = pd.to_datetime(normal_data["Date"])
anomalous_data_1["Date"] = pd.to_datetime(anomalous_data_1["Date"])
anomalous_data_2["Date"] = pd.to_datetime(anomalous_data_2["Date"])
anomalous_data_3["Date"] = pd.to_datetime(anomalous_data_3["Date"])

# Set up the figure and axes for the subplots with constrained layout
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# Plot the normal data on the first subplot
axs[0, 0].plot(normal_data["Date"], normal_data["kWh"], color="blue")
axs[0, 0].set_title("Normal Data")
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel("kWh")

# Plot the anomalous data from the first file
axs[0, 1].plot(anomalous_data_1["Date"], anomalous_data_1["kWh"], color="red")
axs[0, 1].set_title("Anomalous Data 1")
axs[0, 1].set_xlabel("Date")
axs[0, 1].set_ylabel("kWh")

# Plot the anomalous data from the second file
axs[1, 0].plot(anomalous_data_2["Date"], anomalous_data_2["kWh"], color="orange")
axs[1, 0].set_title("Anomalous Data 2")
axs[1, 0].set_xlabel("Date")
axs[1, 0].set_ylabel("kWh")

# Plot the anomalous data from the third file
axs[1, 1].plot(anomalous_data_3["Date"], anomalous_data_3["kWh"], color="green")
axs[1, 1].set_title("Anomalous Data 3")
axs[1, 1].set_xlabel("Date")
axs[1, 1].set_ylabel("kWh")

# Show the plot
plt.show()
