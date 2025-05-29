import os.path
import unittest
from utils.file import find_parent_directory


class MyTestCase(unittest.TestCase):
    def test_find_nearest_folder(self):
        right_data_dir = os.path.join(os.path.dirname(__file__),"data")
        self.assertEqual(find_parent_directory(os.path.dirname(__file__), "data"), right_data_dir)


if __name__ == '__main__':
    unittest.main()
