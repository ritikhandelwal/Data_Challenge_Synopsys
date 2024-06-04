import unittest
import numpy as np
from Data_Challenge_function import extract_features  # Assuming your function is here

class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  
        sr = 22050  
        
        features = extract_features(audio, sr)
        self.assertEqual(len(features), 15)  

if __name__ == "__main__":
  unittest.main()

