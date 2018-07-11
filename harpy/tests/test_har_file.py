"""
Created on Mar 12 09:54:30 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest
import shutil

import numpy as np

from ..har_file import HarFileObj
from ..header_array import HeaderArrayObj

class TestHarFileObj(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_load_from_disk(self):
        hfo = HarFileObj._loadFromDisk(TestHarFileObj._dd + "test.har")
        header_names = hfo.getHeaderArrayNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_write_all(self):
        hfo = HarFileObj._loadFromDisk(TestHarFileObj._dd + "test.har") # Must check loadFromDisk passes test to rely on results from this method
        hfo.writeToDisk("temp.har") # By default, writes all headers

        self.assertTrue(os.path.isfile("temp.har"))

        hfo = HarFileObj._loadFromDisk("temp.har")
        header_names = hfo.getHeaderArrayNames()

        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(header_names, test_hn)]))

        os.remove("temp.har")

    def test_get_header_array_obj(self):
        hfo = HarFileObj._loadFromDisk(
            TestHarFileObj._dd + "test.har")  # Must check loadFromDisk passes test to rely on results from this method
        hao = hfo._getHeaderArrayObjs(["ARR7"])[0]
        self.assertTrue(isinstance(hao, HeaderArrayObj))

    def test_overwrite_header(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_overwrite_header.har")

        hfo = HarFileObj("test_overwrite_header.har")  # Must check loadFromDisk passes test to rely on results from this method

        hao = hfo._getHeaderArrayObjs(["ARR7"])[0]
        hao.array[0,0,0,0,0,0,0] = 42.0

        hfo.writeToDisk("test_overwrite_header.har")

        with self.assertWarns(UserWarning):
            header_names = hfo.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']
        self.assertTrue(all([x == y for (x, y) in zip(header_names, test_hn)]))

        hao = hfo._getHeaderArrayObjs(["ARR7"])[0]
        self.assertTrue(np.isclose(hao.array[0,0,0,0,0,0,0], 42.0))

        os.remove("test_overwrite_header.har")

    def test_addremove_header_array_obj(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_remove_header_array.har")

        hfo = HarFileObj._loadFromDisk("test_remove_header_array.har")
        hao = hfo._removeHeaderArrayObjs("INTA")

        hn = hfo.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'SIMP', 'SIM2', 'NH01', 'ARR7']
        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        hfo._addHeaderArrayObjs("INTA", hao)

        with self.assertRaises(HarFileObj.InvalidHeaderArrayName):
            hfo._addHeaderArrayObjs("TOO LONG NAME", hao)

        hn = hfo.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'SIMP', 'SIM2', 'NH01', 'ARR7', 'INTA']

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        os.remove("test_remove_header_array.har")

    def test_get_real_headerarrays(self):
        shutil.copy2(TestHarFileObj._dd + "test.har", "test_get_real_headerarrays.har")

        hfo = HarFileObj._loadFromDisk("test_get_real_headerarrays.har")
        hn = hfo.getRealHeaderArrayNames()

        test_hn = ['NH01', 'ARR7']

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        os.remove("test_get_real_headerarrays.har")

    def test_get_item(self):

        hfo = HarFileObj._loadFromDisk(TestHarFileObj._dd + "test.har")
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        # Test __getitem __
        XXCDHead=hfo["XXCD"]
        self.assertTrue(isinstance(XXCDHead,HeaderArrayObj))

        # Test case insensitive
        xxcdHead=hfo["xxcd"]
        self.assertTrue(xxcdHead==XXCDHead)

        # Test List get
        HeadList=hfo[test_hn]
        for i,headid in enumerate(test_hn):
            self.assertTrue(hfo[headid] == HeadList[i])

        # Test bad request
        with self.assertRaises(KeyError):
            HeadList = hfo["NOTH"]
        with self.assertRaises(TypeError):
            HeadList = hfo[1]

    def test_del_and_contains(self):
        hfo = HarFileObj._loadFromDisk(TestHarFileObj._dd + "test.har")
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        #test contains + case insensitivity
        self.assertFalse("HNOT" in hfo)
        self.assertTrue( "XXCD" in hfo)
        self.assertTrue("xxcd" in hfo)

        #delete single item
        del hfo["XXCD"]
        self.assertFalse("XXCD" in hfo)

        del hfo["xxcr"]
        self.assertFalse("XXCR" in hfo)

        #test delete list
        del_hn = ['XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']
        del hfo[del_hn]
        self.assertFalse(any([ name in hfo for name in del_hn ]))

    def test_set_item(self):
        hfo = HarFileObj._loadFromDisk(TestHarFileObj._dd + "test.har")
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        xxcdHead = hfo["xxcd"]
        #set item + case insensitivity
        hfo["xxc1"]=xxcdHead
        self.assertTrue("XXC1" in hfo)

        # set item with lists
        hfo[["xxc2","xxc3"]]=[xxcdHead]*2
        self.assertTrue(all(name in hfo for name in ["XXC2","XXC3"]))

        with self.assertRaises(HarFileObj.InvalidHeaderArrayName):
            hfo["TOO LONG NAME"] = xxcdHead

        with self.assertRaises(TypeError):
            hfo[1] = xxcdHead



if __name__ == "__main__":
    unittest.main()
