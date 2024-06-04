import unittest
import numpy as np
from Data_Challenge_function import extract_features  # Assuming your notebook is named 'Data_Challenge'

class TestFeatureExtraction(unittest.TestCase):
  def test_extract_features(self):
    # Dummy audio signal for testing
    audio = np.random.random(22050)
    sr = 22050
    features = extract_features(audio, sr)

    # Expected output shape (13,) for 13 MFCCs, zero-crossing rate, and spectral centroid
    self.assertEqual(features.shape, (13,))

if __name__ == '__main__':
  unittest.main()
