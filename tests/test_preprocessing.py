"""Test preprocessing"""
import unittest
import numpy as np
from src.preprocessing import preprocess_image


class TestPreprocessing(unittest.TestCase):
    
    def test_preprocess_array(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        processed = preprocess_image(img)
        
        self.assertEqual(processed.shape, (32, 32, 3))
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))


if __name__ == '__main__':
    unittest.main()