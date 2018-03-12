"""
Created on Mar 12 09:54:30 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest
import shutil
import sys

import numpy as np

from harpy.har_file import HarFileObj

# print(sys.path)
# sys.path.insert(0, "/opt/anaconda3/lib/python3.5/site-packages")
# print(sys.path)

from harpy.HAR import HAR as _HAR
from harpy.HAR_IO import HAR_IO as _HARIO

class TestHarFileObj(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_load_from_disk(self):
        hfo = HarFileObj.loadFromDisk(TestHarFileObj._dd + "test.har")
        header_names = hfo["hfio"].getHeaderNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_write_all(self):
        hfo = HarFileObj.loadFromDisk(TestHarFileObj._dd + "test.har") # Must check loadFromDisk passes test to rely on results from this method
        hfo.writeToDisk("temp.har") # By default, writes all headers

        self.assertTrue(os.path.isfile("temp.har"))

        hfo = HarFileObj.loadFromDisk("temp.har")
        # print(hfo)
        header_names = hfo["hfio"].getHeaderNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(header_names, test_hn)]))

        os.remove("temp.har")

if __name__ == "__main__":
    unittest.main()
