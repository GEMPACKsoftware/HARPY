"""
Created on Mar 13 09:19:42 2018

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


class TestHeaderArray(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_is_valid(self): # Relies on external functions operating correctly...
        hfo = HarFileObj.loadFromDisk(TestHeaderArray._dd + "test.har")
        haos = hfo.getHeaderArrayObjs()
        for hao in haos:
            self.assertTrue(hao.is_valid())

    def test_header_array_from_data(self):

        # Test data
        array_1d = np.array([1.0, 2.0, 3.0])
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        array_7d = np.array(list(range(2**7)))
        array_7d.reshape((2, 2, 2, 2, 2, 2, 2))

        with self.assertRaises(HeaderArrayObj.InvalidHeaderArrayName):
            HeaderArrayObj.HeaderArrayFromData(name="NAME_THAT_IS_TOO_LONG", array=array_1d)

        hao = HeaderArrayObj.HeaderArrayFromData(name="ARR1", array=array_1d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

        hao = HeaderArrayObj.HeaderArrayFromData(name="ARR2", array=array_2d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

        hao = HeaderArrayObj.HeaderArrayFromData(name="ARR7", array=array_7d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

    def test_array_operation(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(name="ARRA", array=array_2d)
        hao2 = HeaderArrayObj.HeaderArrayFromData(name="ARRB", array=array_2d)

        hao3 = hao1 + hao2

        self.assertTrue(np.allclose(hao3["array"], array_2d*2))

        hao3 = hao1 - hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[0, 0], [0, 0]])))

        hao3 = hao1 * hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 4], [9, 16]])))

        hao3 = hao1 / hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 1], [1, 1]])))



if __name__ == "__main__":
    unittest.main()
