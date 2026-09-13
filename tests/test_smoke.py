import unittest

import numpy as np

from src.dataset import fourier_features, normalized_coordinates
from src.model import TemporalVideoField


class INRSmokeTest(unittest.TestCase):
    def test_coordinate_encoding(self):
        coordinates = normalized_coordinates(2, 3, 4)
        self.assertEqual(coordinates.shape, (24, 3))
        self.assertEqual(fourier_features(coordinates, 6).shape, (24, 36))

    def test_temporal_interpolation(self):
        images = np.stack([np.zeros((2, 2, 3)), np.ones((2, 2, 3))]).astype(np.float32)
        masks = np.stack([np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32)
        model = TemporalVideoField(np.asarray([0, 2]), images, masks)
        image, mask = model.predict_frame(1)
        self.assertTrue(np.allclose(image.numpy(), 0.5))
        self.assertTrue(np.allclose(mask.numpy(), 0.5))


if __name__ == "__main__":
    unittest.main()
