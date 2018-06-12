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
from harpy.header_array import HeaderArrayObj

class TestHarFileObj(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_load_from_disk(self):
        hfo = HarFileObj.loadFromDisk(TestHarFileObj._dd + "test.har")
        header_names = hfo.getHeaderArrayNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_write_all(self):
        hfo = HarFileObj.loadFromDisk(TestHarFileObj._dd + "test.har") # Must check loadFromDisk passes test to rely on results from this method
        hfo.writeToDisk("temp.har") # By default, writes all headers

        self.assertTrue(os.path.isfile("temp.har"))

        hfo = HarFileObj.loadFromDisk("temp.har")
        header_names = hfo.getHeaderArrayNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(header_names, test_hn)]))

        os.remove("temp.har")

    def test_get_header_array_obj(self):
        hfo = HarFileObj.loadFromDisk(
            TestHarFileObj._dd + "test.har")  # Must check loadFromDisk passes test to rely on results from this method
        hao = hfo.getHeaderArrayObjs(["ARR7"])[0]
        self.assertTrue(isinstance(hao, HeaderArrayObj))

    def test_overwrite_header(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_overwrite_header.har")

        hfo = HarFileObj.loadFromDisk("test_overwrite_header.har")  # Must check loadFromDisk passes test to rely on results from this method

        hao = hfo.getHeaderArrayObjs(["ARR7"])[0]
        hao["array"][0,0,0,0,0,0,0] = 42.0

        hfo.writeToDisk("test_overwrite_header.har")

        hfo = HarFileObj.loadFromDisk("test_overwrite_header.har")
        header_names = hfo.getHeaderArrayNames()
        # header_names = hfo["hfio"].getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']
        self.assertTrue(all([x == y for (x, y) in zip(header_names, test_hn)]))

        hao = hfo.getHeaderArrayObjs(["ARR7"])[0]
        self.assertTrue(np.isclose(hao["array"][0,0,0,0,0,0,0], 42.0))

        os.remove("test_overwrite_header.har")

    def test_addremove_header_array_obj(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_remove_header_array.har")

        hfo = HarFileObj.loadFromDisk("test_remove_header_array.har")
        hao = hfo.removeHeaderArrayObjs("INTA")

        hn = hfo.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        hfo.addHeaderArrayObjs(hao)

        hn = hfo.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'SIMP', 'SIM2', 'NH01', 'ARR7', 'INTA']

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        os.remove("test_remove_header_array.har")

    def test_get_real_headerarrays(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_get_real_headerarrays.har")

        hfo = HarFileObj.loadFromDisk("test_get_real_headerarrays.har")
        hn = hfo.getRealHeaderArrayNames()

        test_hn = ['NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        os.remove("test_get_real_headerarrays.har")

if __name__ == "__main__":
    unittest.main()
