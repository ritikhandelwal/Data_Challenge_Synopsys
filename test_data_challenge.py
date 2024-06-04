import unittest
from Data_Challenge_function import extract_features  # Assuming your function is here

class TestFeatureExtraction(unittest.TestCase):

  def test_empty_data(self):
    """Test if the function handles empty audio data."""
    default_sr = 22050  # Example default sample rate
    features = extract_features(None, default_sr)  # Pass None and default sr

    # Assert expected behavior for empty data
    self.assertIsInstance(features, dict)  # Check if it returns a dictionary
    self.assertEqual(features, {})  # Check if the dictionary is empty

if __name__ == "__main__":
  unittest.main()

