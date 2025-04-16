# 🌞 Solar Panel Anomaly Detection & Forecasting

This project builds an anomaly detection and forecasting system for solar panel output using a deep learning autoencoder. It uses clean data from a reference panel to train the model, then forecasts and detects anomalies in three test panels.

---

## 📌 Features

- ✅ **Data Generation** – Synthetic normal and anomalous data creation.
- ✅ **Model Training** – Autoencoder model built with TensorFlow/Keras.
- ✅ **Forecasting** – Predicts future solar panel output.
- ✅ **Anomaly Detection** – Flags abnormal patterns and outliers.
- ✅ **Visualization** – Highlights anomalies and prediction boundaries.

---

## 🧠 How It Works

1. **Reference Panel (`solar_output.csv`)** – Clean solar data used to train the autoencoder.
2. **Test Panels** – `solar_output_with_varied_anomalies_*.csv` contain data with artificial anomalies.
3. **Training** – The autoencoder learns the patterns in the reference data.
4. **Forecasting** – Test data is passed through the trained model to predict expected output.
5. **Anomaly Detection** – Any point with prediction error beyond a threshold is flagged.
6. **Plotting** – Graphs show both normal and anomalous behavior across all test datasets.

---

## 🗂 Project Structure

```plaintext
project-root/
│
├── data/
│   ├── solar_output.csv                           # Clean reference data
│   ├── solar_output_with_varied_anomalies_1.csv   # Test panel with anomaly type 1
│   ├── solar_output_with_varied_anomalies_2.csv   # Test panel with anomaly type 2
│   ├── solar_output_with_varied_anomalies_3.csv   # Test panel with anomaly type 3
│
├── scaler.pkl                       # Skaler for autoencoder model file
├── autoencoder_model.keras          # Trained autoencoder model file
│
├── detector.py                      # Trains and applies the model to detect anomalies
├── forecast.py                      # Forecasts future output using the model
├── generator_model.py               # Defines the model architecture
├── generator_normal_data.py         # Creates synthetic clean data
├── generator_unnormal_data.py       # Creates synthetic data with anomalies
├── graph_data.py                    # Plots outputs, predictions, and anomalies
├── graph_data.py                    # Tests plots are created correctly and anomalies are correctly detected based
│
├── README.md                        # 📘 Project documentation (this file)
├── requirements.txt                 # 📦 Python dependencies
    
```

## ⚙️ Installation
1. **Clone the Repository**
```plaintext
git clone https://github.com/nen0n/solar-panel-anomaly-detector.git
cd solar-panel-anomaly-detector
```

2. **Create and Activate a Virtual Environment**

```plaintext
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```plaintext
pip install -r requirements.txt
```

## 🚀 Usage

1. **Generate Data (Optional)**

You can generate synthetic datasets:

```plaintext
python generator_normal_data.py       # Generates clean reference panel data
python generator_unnormal_data.py     # Generates test panels with varied anomalies
```

2. **Train and Detect Anomalies.**

Train the model on clean data and detect anomalies in test datasets:

```plaintext
python detector.py
```
3. **Forecast Future Solar Output.**

Use the trained model to forecast future solar panel behavior:

```plaintext
python forecast.py
```

4. **Plot Results**

Generate subplots comparing normal and anomalous behavior:
```plaintext
python graph_data.py
```

5. **Testing**

Run all the test cases and show the results in your terminal.
```plaintext
python test_anomalies.py
```

##  📊 Visual Output
The system produces multiple visualizations, including:

- **Normal Panel Output** – Clean baseline data.

- **Test Panels** – With anomalies clearly marked.

- **Forecasts** – Predicted vs. actual output.

- **Anomaly Highlighting** – Red markers for detected deviations.

- **Boundaries** – Visualization of prediction tolerance windows.

Each test panel is plotted separately in subplots for better comparison.

##  ⚙️ Configuration
You can modify the following to tune the system:

- **Anomaly detection threshold**

- **Forecast window size**

- **Autoencoder layer sizes and activation functions**

- **Dataset file paths and lengths**

##  📦 Requirements
- **Python 3.12**

- **numpy**

- **pandas**

- **matplotlib**

- **tensorflow**

- **scikit-learn**

- **joblib**

- **pytest**
- **pylint**

Install all dependencies with:

```plaintext
pip install -r requirements.txt
```