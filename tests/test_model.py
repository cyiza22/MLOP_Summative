"""Test model"""
import unittest
from src.model import create_model


class TestModel(unittest.TestCase):
    
    def test_create_model(self):
        model = create_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))


if __name__ == '__main__':
    unittest.main()