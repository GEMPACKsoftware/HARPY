"""
Created on Mar 02 11:17:41 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest
import shutil
import sys

import numpy as np

from harpy.har_file_io import HarFileIO

# print(sys.path)
# sys.path.insert(0, "/opt/anaconda3/lib/python3.5/site-packages")
# print(sys.path)

class TestHarFileIO(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_read_har_file_info(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
        # header_names = list(hfi["headers"].keys())
        header_names = [ha_info["name"] for ha_info in hfi["ha_infos"]]
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']

        self.assertTrue(all([x == y for (x,y) in zip(header_names, test_hn)]))

    def test_read_1C(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
        header = HarFileIO.readHeader(hfi, "SIMP")
        self.assertTrue(all([x == y for (x, y) in zip(header["array"], ["A", "B"])]))

    def test_read_1C_2(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test_read.har")
        header = HarFileIO.readHeader(hfi, "CHST")
        test_array = ["F_string    ", "B_string    ", "C_string    ", "D_string    ", "E_string    "]
        self.assertTrue(all([x == y for (x, y) in zip(header["array"], test_array)]))

    def test_read_RE(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
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
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
        nh01header = HarFileIO.readHeader(hfi, "NH01")

        self.assertTrue(np.isclose(nh01header["array"][0, 0], 1.0))
        self.assertTrue(np.isclose(nh01header["array"][0, 1], 3.14))
        self.assertTrue(np.isclose(nh01header["array"][1, 0], 3.14))
        self.assertTrue(np.isclose(nh01header["array"][1, 1], 5.0))

    def test_read_2I(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
        intaheader = HarFileIO.readHeader(hfi, "INTA")

        self.assertEqual(intaheader["array"][0, 0], 0)
        self.assertEqual(intaheader["array"][0, 1], 1)
        self.assertEqual(intaheader["array"][1, 0], 4)
        self.assertEqual(intaheader["array"][3, 3], 15)

    def test_write_1C(self):
        shutil.copy2(TestHarFileIO._dd + "test.har", "temp.har")

        hfi = HarFileIO.readHarFileInfo("temp.har")
        chst_header = HarFileIO.readHeader(hfi, "CHST")
        self.assertTrue(chst_header.is_valid())

        chst_header["array"][0] = "F_string" # Change one element
        HarFileIO.writeHeaders("temp.har", chst_header)

        hfi = HarFileIO.readHarFileInfo("temp.har")
        test_hn = ['CHST']
        hfi_headers = [ha_info["name"] for ha_info in hfi["ha_infos"]]
        self.assertTrue(all([x == y for (x, y) in zip(hfi_headers, test_hn)]))

        chst_header = HarFileIO.readHeader(hfi, "CHST")
        test_array = ["F_string    ", "B_string    ", "C_string    ", "D_string    ", "E_string    "]

        self.assertTrue(all([x == y for (x, y) in zip(chst_header["array"], test_array)]))

        os.remove("temp.har")

    def test_write_2I(self):
        shutil.copy2(TestHarFileIO._dd + "test.har", "temp.har")

        hfi = HarFileIO.readHarFileInfo("temp.har")
        inta_header = HarFileIO.readHeader(hfi, "INTA")
        self.assertTrue(inta_header.is_valid())

        inta_header["array"][3,3] = 30
        HarFileIO.writeHeaders("temp.har", inta_header)

        hfi = HarFileIO.readHarFileInfo("temp.har")
        inta_header = HarFileIO.readHeader(hfi, "INTA")

        self.assertEqual(inta_header["array"][3, 3], 30)

        os.remove("temp.har")

    def test_write_RE(self):
        shutil.copy2(TestHarFileIO._dd + "test.har", "temp.har")

        hfi = HarFileIO.readHarFileInfo("temp.har")
        arr7_header = HarFileIO.readHeader(hfi, "ARR7")
        self.assertTrue(arr7_header.is_valid())

        arr7_header["array"][1, 1, 1, 1, 1, 1, 1] = 103.14
        HarFileIO.writeHeaders("temp.har", arr7_header)

        hfi = HarFileIO.readHarFileInfo("temp.har")
        inta_header = HarFileIO.readHeader(hfi, "ARR7")

        self.assertTrue(np.isclose(inta_header["array"][1, 1, 1, 1, 1, 1, 1], 103.14))

        os.remove("temp.har")

    def test_get_header_names(self):
        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")
        hn = hfi.getHeaderArrayNames()
        test_hn = ['XXCD', 'XXCR', 'XXCP', 'XXHS', 'CHST', 'INTA', 'SIMP', 'SIM2', 'NH01', 'ARR7']
        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

    def test_write_all(self):
        # shutil.copy2(TestHarFileIO._dd + "test.har", "temp.har")

        hfi = HarFileIO.readHarFileInfo(TestHarFileIO._dd + "test.har")

        nh01 = HarFileIO.readHeader(hfi, "NH01")
        arr7 = HarFileIO.readHeader(hfi, "ARR7")

        HarFileIO.writeHeaders("temp.har", [nh01, arr7])

        hfi = HarFileIO.readHarFileInfo("temp.har")
        hn = hfi.getHeaderArrayNames()
        test_hn = ["NH01", "ARR7"]

        self.assertTrue(all([x == y for (x, y) in zip(hn, test_hn)]))

        os.remove("temp.har")

if __name__ == "__main__":
    unittest.main()
