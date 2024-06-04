import unittest
import numpy as np
from Data_Challenge_function import extract_features  # Assuming your function is here

class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        # Create mock audio data
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Convert list to numpy array
        sr = 22050  # Example sampling rate
        
        features = extract_features(audio, sr)
        
        # Assert that the output has the expected shape or values
        self.assertEqual(len(features), 15)  # Ensure correct number of features
        # Add more assertions as needed

if __name__ == "__main__":
  unittest.main()

