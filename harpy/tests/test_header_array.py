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



        hao = HeaderArrayObj.HeaderArrayFromData(array=array_1d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

        hao = HeaderArrayObj.HeaderArrayFromData(array=array_2d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

        hao = HeaderArrayObj.HeaderArrayFromData(array=array_7d)
        self.assertTrue(hao.is_valid())
        # TODO: Include array type-checking

    def test_array_operation(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData( array=array_2d)
        hao2 = HeaderArrayObj.HeaderArrayFromData( array=array_2d)

        hao3 = hao1 + hao2

        self.assertTrue(np.allclose(hao3["array"], array_2d*2))

        hao3 = hao1 - hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[0, 0], [0, 0]])))

        hao3 = hao1 * hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 4], [9, 16]])))

        hao3 = hao1 / hao2

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 1], [1, 1]])))

    def test_array_operation_ndarray(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + array_2d

        self.assertTrue(np.allclose(hao3["array"], array_2d*2))

        hao3 = hao1 - array_2d

        self.assertTrue(np.allclose(hao3["array"], np.array([[0, 0], [0, 0]])))

        hao3 = hao1 * array_2d

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 4], [9, 16]])))

        hao3 = hao1 / array_2d

        self.assertTrue(np.allclose(hao3["array"], np.array([[1, 1], [1, 1]])))

    def test_array_operation_int(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + 2

        self.assertTrue(np.allclose(hao3["array"], array_2d + 2))

        hao3 = hao1 - 1

        self.assertTrue(np.allclose(hao3["array"], array_2d - 1))

        hao3 = hao1 * 3

        self.assertTrue(np.allclose(hao3["array"], array_2d * 3))

        hao3 = hao1 / 2

        self.assertTrue(np.allclose(hao3["array"], array_2d / 2))

    def test_array_operation_float(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + 1.5

        self.assertTrue(np.allclose(hao3["array"], array_2d + 1.5))

        hao3 = hao1 - 0.5

        self.assertTrue(np.allclose(hao3["array"], array_2d - 0.5))

        hao3 = hao1 * 3.14

        self.assertTrue(np.allclose(hao3["array"], array_2d * 3.14))

        hao3 = hao1 / -2.5

        self.assertTrue(np.allclose(hao3["array"], array_2d / -2.5))

    def test_attributes_style(self):
        """Note that the values set by the setter methods are NOT guaranteed to be consistent or legitimate. Implementation of further checks with the attribute-style referencing may cause this test to fail."""

        hao = HeaderArrayObj()

        # Test setters
        hao.name = "ABC"
        hao.coeff_name = "ABCDEF"
        hao.long_name = "A test header array object."
        hao.array = np.array([[1.0, 2.0], [3.0, 4.0]])
        hao.data_type = "2R"
        hao.version = 1
        hao.storage_type = "SPSE"
        hao.file_dims = 2
        hao.sets = ["A"]

        # Test getters
        self.assertEqual(hao.name, "ABC")
        self.assertEqual(hao.coeff_name, "ABCDEF")
        self.assertEqual(hao.long_name, "A test header array object.")
        self.assertTrue(np.allclose(hao.array, np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertEqual(hao.data_type, "2R")
        self.assertEqual(hao.version, 1)
        self.assertEqual(hao.storage_type, "SPSE")
        self.assertEqual(hao.file_dims, 2)
        self.assertEqual(hao.sets, ["A"])


if __name__ == "__main__":
    unittest.main()
