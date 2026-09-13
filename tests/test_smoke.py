import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from dataset import normalized_coordinates, fourier_features

class INRSmokeTest(unittest.TestCase):
    def test_coordinates_and_encoding(self):
        coords = normalized_coordinates(2, 3, 4)
        self.assertEqual(coords.shape, (24, 3)); self.assertTrue((abs(coords) <= 1).all())
        self.assertEqual(fourier_features(coords, 6).shape, (24, 36))

if __name__ == '__main__': unittest.main()
