"""
Created on Mar 14 13:05:03 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
import unittest

from harpy.tests.test_har_file import TestHarFileObj
from harpy.tests.test_har_file_io import TestHarFileIO
from harpy.tests.test_header_array import TestHeaderArray

class TestHarpy(unittest.TestSuite):
    """TestConCERO is a unittest.TestSuite, which provides an easy access point to run all the tests.
    """

    def __init__(self):
        super().__init__()
        for testcase in (TestHarFileIO, TestHarFileObj, TestHeaderArray):
            self.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(testcase))

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestHarpy())