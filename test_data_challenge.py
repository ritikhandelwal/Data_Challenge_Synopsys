import unittest
from Data_Challenge_function import extract_features  # Assuming your function is here

class TestFeatureExtraction(unittest.TestCase):

  def test_extract_features(self):
        # Create mock audio data
        audio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example audio data
        sr = 22050  # Example sampling rate
        
        features = extract_features(audio, sr)
        
        # Assert that the output has the expected shape or values
        self.assertEqual(len(features), 15)  # Ensure correct number of features
        # Add more assertions as needed

if __name__ == "__main__":
  unittest.main()

