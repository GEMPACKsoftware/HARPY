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
        hfo = HarFileObj._loadFromDisk(TestHeaderArray._dd + "test.har")
        haos = hfo._getHeaderArrayObjs()
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

        self.assertTrue(np.allclose(hao3.array, array_2d*2))

        hao3 = hao1 - hao2

        self.assertTrue(np.allclose(hao3.array, np.array([[0, 0], [0, 0]])))

        hao3 = hao1 * hao2

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 4], [9, 16]])))

        hao3 = hao1 / hao2

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 1], [1, 1]])))

    def test_array_operation_ndarray(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + array_2d

        self.assertTrue(np.allclose(hao3.array, array_2d*2))

        hao3 = hao1 - array_2d

        self.assertTrue(np.allclose(hao3.array, np.array([[0, 0], [0, 0]])))

        hao3 = hao1 * array_2d

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 4], [9, 16]])))

        hao3 = hao1 / array_2d

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 1], [1, 1]])))

        hao3 = array_2d + hao1

        self.assertTrue(np.allclose(hao3.array, array_2d*2))

        hao3 = array_2d - hao1

        self.assertTrue(np.allclose(hao3.array, np.array([[0, 0], [0, 0]])))

        hao3 = array_2d * hao1

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 4], [9, 16]])))

        hao3 = array_2d / hao1

        self.assertTrue(np.allclose(hao3.array, np.array([[1, 1], [1, 1]])))

    def test_array_operation_int(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + 2

        self.assertTrue(np.allclose(hao3.array, array_2d + 2))

        hao3 = hao1 - 1

        self.assertTrue(np.allclose(hao3.array, array_2d - 1))

        hao3 = hao1 * 3

        self.assertTrue(np.allclose(hao3.array, array_2d * 3))

        hao3 = hao1 / 2

        self.assertTrue(np.allclose(hao3.array, array_2d / 2))

        hao3 = 2 + hao1

        self.assertTrue(np.allclose(hao3.array, array_2d + 2))

        hao3 = 1 - hao1

        self.assertTrue(np.allclose(hao3.array, 1 - array_2d ))

        hao3 = 3 * hao1

        self.assertTrue(np.allclose(hao3.array, array_2d * 3))

        hao3 = 2 / hao1

        self.assertTrue(np.allclose(hao3.array, 2 / array_2d ))

    def test_array_operation_float(self):

        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        hao1 = HeaderArrayObj.HeaderArrayFromData(array=array_2d)

        hao3 = hao1 + 1.5

        self.assertTrue(np.allclose(hao3.array, array_2d + 1.5))

        hao3 = hao1 - 0.5

        self.assertTrue(np.allclose(hao3.array, array_2d - 0.5))

        hao3 = hao1 * 3.14

        self.assertTrue(np.allclose(hao3.array, array_2d * 3.14))

        hao3 = hao1 / -2.5

        self.assertTrue(np.allclose(hao3.array, array_2d / -2.5))

        hao3 = 1.5 + hao1

        self.assertTrue(np.allclose(hao3.array, array_2d + 1.5))

        hao3 = 0.5 - hao1

        self.assertTrue(np.allclose(hao3.array, 0.5 - array_2d))

        hao3 = 3.14 * hao1

        self.assertTrue(np.allclose(hao3.array, array_2d * 3.14))

        hao3 = -2.5 / hao1

        self.assertTrue(np.allclose(hao3.array, -2.5 / array_2d))

    def test_attributes_style(self):
        """Note that the values set by the setter methods are NOT guaranteed to be consistent or legitimate. Implementation of further checks with the attribute-style referencing may cause this test to fail."""

        hao = HeaderArrayObj()

        # Test setters
        hao.name = "ABC"
        hao.coeff_name = "ABCDEF"
        hao.long_name = "A test header array object."
        hao.array = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Test getters
        self.assertEqual(hao.name, "ABC")
        self.assertEqual(hao.coeff_name, "ABCDEF")
        self.assertEqual(hao.long_name, "A test header array object.")
        self.assertTrue(np.allclose(hao.array, np.array([[1.0, 2.0], [3.0, 4.0]])))

    def test_getitem_setitem(self):
        hfo = HarFileObj._loadFromDisk(TestHeaderArray._dd + "test.har")
        nh01=hfo["NH01"]

        nh01[:,:]=np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(nh01.array, np.array([[1.0, 2.0], [3.0, 4.0]])))

        nh01[[0,1], [0,1]] = np.array([[4.0, 3.0], [2.0, 1.0]])
        self.assertTrue(np.allclose(nh01.array, np.array([[4.0, 3.0], [2.0, 1.0]])))

        nh01[ 0, : ] = np.array([1.0, 2.0])
        self.assertTrue(np.allclose(nh01.array, np.array([[1.0, 2.0], [2.0, 1.0]])))

        nh01[ [0], : ] = np.array([3.0, 4.0])
        self.assertTrue(np.allclose(nh01.array, np.array([[3.0, 4.0], [2.0, 1.0]])))

        nh01[ ... ]=np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(nh01.array, np.array([[1.0, 2.0], [3.0, 4.0]])))

        nh01[["A","B"], [0,1]] = np.array([[4.0, 3.0], [2.0, 1.0]])
        self.assertTrue(np.allclose(nh01.array, np.array([[4.0, 3.0], [2.0, 1.0]])))

        newHead=nh01[["A","B"], [0,1]]
        self.assertTrue(np.allclose(newHead.array, np.array([[4.0, 3.0], [2.0, 1.0]])))
        self.assertTrue(newHead.setNames==nh01.setNames)

        newHead=nh01[["A"], [0,1]]
        self.assertTrue(np.allclose(newHead.array, np.array([[4.0, 3.0]])))
        self.assertTrue(newHead.rank==2)

        newHead=nh01["A", [0,1]]
        self.assertTrue(np.allclose(newHead.array, np.array([4.0, 3.0])))
        self.assertTrue(newHead.rank==1)

        newHead=nh01["A", 0]
        self.assertTrue(np.allclose(newHead.array, 4.0 ))
        self.assertTrue(newHead.rank==0)

        newHead2=newHead[None]
        self.assertTrue(newHead2.rank == 1)


        newHead=nh01["A", [0,1]] #[4,3]
        newHead2=newHead[:,None]*nh01 # multiply as col vector
        self.assertTrue(np.allclose(newHead2.array, np.array([[4.0],[3.0]])*np.array([[4.0, 3.0], [2.0, 1.0]])))

        newHead2=newHead[None,[0,1]]*nh01 # multiply as row vector
        self.assertTrue(np.allclose(newHead2.array, np.array([[4.0,3.0]])*np.array([[4.0, 3.0], [2.0, 1.0]])))


if __name__ == "__main__":
    unittest.main()
