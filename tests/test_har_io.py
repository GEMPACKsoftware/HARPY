"""
Created on Mar 02 11:17:41 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest

from harpy.har_io import HAR_IO

from harpy.HAR import HAR as _HAR
from harpy.HAR_IO import HAR_IO as _HARIO

class TestHarIO(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_read_har_file_info(self):
        hfi = HAR_IO.readHarFileInfo(TestHarIO._dd + "test.har")
        header_names = list(hfi["headers"].keys())
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_read_header_names(self):
        header_names = HAR_IO.readHeaderNames(TestHarIO._dd + "test.har")
        test_names = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_names)]))

    def test_read_1C(self):
        hfi = HAR_IO.readHarFileInfo(TestHarIO._dd + "test.har")
        print(hfi)

        har_obj = _HAR(TestHarIO._dd + "test.har", "r")
        print(type(har_obj))
        header = har_obj.getHeader("ARR7")
        print(type(header))

        # print(har_obj.getHeader("SIMP"))

        HAR_IO.readHeader(hfi, "ARR7")

        print(hfi["headers"]["ARR7"]["array"])


if __name__ == "__main__":
    unittest.main()
