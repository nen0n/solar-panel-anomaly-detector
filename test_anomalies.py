import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from unittest.mock import patch, MagicMock
from detector import detect_anomalies, process_file


class TestAnomalyDetection(unittest.TestCase):

    def test_detect_empty_array(self):
        data = np.array([])
        anomalies = detect_anomalies(data, threshold=6)
        self.assertEqual(len(anomalies), 0)

    @patch("detector.plt.show")
    @patch("detector.load_model")
    def test_process_file_mocked(self, mock_load_model, mock_show):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(100, 1)
        mock_load_model.return_value = mock_model

        test_csv = "dummy_test.csv"
        df = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 100))})
        df.to_csv(test_csv, index=False)

        try:
            process_file(test_csv, mock_model, threshold=0.4, idx=0)
            mock_model.predict.assert_called()
            mock_show.assert_called()
        finally:
            import os
            os.remove(test_csv)

    def test_pandas_dataframe_creation(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

    def test_pandas_csv_roundtrip(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        df.to_csv("temp_test.csv", index=False)
        loaded = pd.read_csv("temp_test.csv")
        self.assertTrue(df.equals(loaded))
        import os
        os.remove("temp_test.csv")

    def test_matplotlib_scatter_plot(self):
        fig, ax = plt.subplots()
        scatter = ax.scatter([1, 2, 3], [3, 2, 1])
        self.assertEqual(scatter.get_offsets().shape[0], 3)
        plt.close(fig)

    def test_tensorflow_basic_tensor(self):
        t = tf.constant([1.0, 2.0, 3.0])
        self.assertIsInstance(t, tf.Tensor)
        self.assertEqual(tf.reduce_sum(t).numpy(), 6.0)

    def test_keras_model_prediction(self):
        model = keras.Sequential([
            layers.Dense(1, input_shape=(1,))
        ])
        model.compile(optimizer='adam', loss='mse')
        result = model.predict(np.array([[1.0], [2.0]]), verbose=0)
        self.assertEqual(result.shape, (2, 1))

    def test_keras_training(self):
        x = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([[2], [4], [6], [8]], dtype=float)

        model = keras.Sequential([
            layers.Dense(units=1, input_shape=[1])
        ])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        history = model.fit(x, y, epochs=5, verbose=0)

        self.assertIn('loss', history.history)
        self.assertEqual(len(history.history['loss']), 5)

    def test_pandas_filtering(self):
        df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
        filtered = df[df['val'] > 3]
        self.assertEqual(filtered.shape[0], 2)
        self.assertListEqual(filtered['val'].tolist(), [4, 5])

    def test_tensorflow_model_save_and_load(self):
        model = keras.Sequential([
            layers.Dense(1, input_shape=(1,))
        ])
        model.compile(optimizer='adam', loss='mse')
        save_path = "temp_model.keras"
        model.save(save_path)
        loaded_model = keras.models.load_model(save_path)
        prediction = loaded_model.predict(np.array([[1.0]]), verbose=0)
        self.assertEqual(prediction.shape, (1, 1))

        import shutil
        shutil.rmtree(save_path, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
