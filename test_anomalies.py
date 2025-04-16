import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd
from detector import detect_anomalies, process_file


class TestAnomalyDetection(unittest.TestCase):

    def test_detect_anomalies_no_anomalies(self):
        data = np.array([5, 5.2, 5.1, 5, 5.05])
        anomalies = detect_anomalies(data, threshold=0.5, window_size=2)
        self.assertFalse(any(anomalies))

    @patch("detector.plt.show")
    @patch("detector.load_model")
    def test_process_file_mocked(self, mock_load_model, mock_show):
        # Mock model and its predict function
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(100, 1)
        mock_load_model.return_value = mock_model

        # Create a dummy CSV file
        test_csv = "dummy_test.csv"
        df = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 100))})
        df.to_csv(test_csv, index=False)

        try:
            process_file(test_csv, mock_model, threshold=0.4, idx=0)
            mock_model.predict.assert_called()
            mock_show.assert_called()  # Ensure plot.show() was called
        finally:
            import os

            os.remove(test_csv)


if __name__ == "__main__":
    unittest.main()
