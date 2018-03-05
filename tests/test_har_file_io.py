"""
Created on Mar 02 11:17:41 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest

import numpy as np

from harpy.har_file_io import HarFileIO

from harpy.HAR import HAR as _HAR
from harpy.HAR_IO import HAR_IO as _HARIO

class TestHarIO(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_read_har_file_info(self):
        hfi = HarFileIO.readHarFileInfo(TestHarIO._dd + "test.har")
        header_names = list(hfi["headers"].keys())
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_read_header_names(self):
        header_names = HarFileIO.readHeaderNames(TestHarIO._dd + "test.har")
        test_names = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_names)]))

    def test_read_1C(self):
        hfi = HarFileIO.readHarFileInfo(TestHarIO._dd + "test.har")
        header = HarFileIO.readHeader(hfi, "SIMP")
        self.assertTrue(all(x == y for (x, y) in zip(header["array"], ["A", "B"])))

    def test_read_RE(self):
        hfi = HarFileIO.readHarFileInfo(TestHarIO._dd + "test.har")
        arr7header = HarFileIO.readHeader(hfi, "ARR7")

        self.assertTrue(np.isclose(arr7header["array"][0, 0, 0, 0, 0, 0, 0], 1.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 0, 0, 0, 0, 0, 1], 2.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 0, 0, 0, 0, 1, 0], 3.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 0, 0, 0, 1, 0, 0], 4.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 0, 0, 1, 0, 0, 0], 5.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 0, 1, 0, 0, 0, 0], 6.0))
        self.assertTrue(np.isclose(arr7header["array"][0, 1, 0, 0, 0, 0, 0], 7.0))
        self.assertTrue(np.isclose(arr7header["array"][1, 0, 0, 0, 0, 0, 0], 8.0))
        self.assertTrue(np.isclose(arr7header["array"][1, 1, 1, 1, 1, 1, 1], 9.0))

    def test_read_2D_RE(self):
        hfi = HarFileIO.readHarFileInfo(TestHarIO._dd + "test.har")
        nh01header = HarFileIO.readHeader(hfi, "NH01")

        self.assertTrue(np.isclose(nh01header["array"][0, 0], 1.0))
        self.assertTrue(np.isclose(nh01header["array"][0, 1], 3.14))
        self.assertTrue(np.isclose(nh01header["array"][1, 0], 3.14))
        self.assertTrue(np.isclose(nh01header["array"][1, 1], 5.0))

    def test_read_2I(self):
        hfi = HarFileIO.readHarFileInfo(TestHarIO._dd + "test.har")
        intaheader = HarFileIO.readHeader(hfi, "INTA")

        self.assertEqual(intaheader["array"][0, 0], 0)
        self.assertEqual(intaheader["array"][0, 1], 1)
        self.assertEqual(intaheader["array"][1, 0], 4)
        self.assertEqual(intaheader["array"][3, 3], 15)



if __name__ == "__main__":
    unittest.main()
