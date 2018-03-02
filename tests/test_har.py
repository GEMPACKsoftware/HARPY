"""
Tests the read HAR functionality of HARPY. Class has been written for Python 3, and no effort has been made to make \
this class Python 2 compatible.

Created on Jan 29 11:57:37 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest

#import harpy.HAR
import harpy.har as har

class TestHar(unittest.TestCase):

    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_readheader(self):
        har = har.HAR(TestHar._dd + "Mdatnew7.har", "r")
        h = har.getHeader("MAR1") # HARPY used to thrown exception at this point for Python 3

    def test_get_file_name(self):
        har = har.HAR(TestHar._dd + "Mdatnew7.har", "r")
        self.assertEqual(os.path.relpath(har.getFileName()),
                         os.path.join("tests", "testdata", "Mdatnew7.har"))

    def test_remove_header(self):
        har = har.HAR(TestHar._dd + "Mdatnew7.har", "r")
        har.removeHeader("B015")
        self.assertEqual("B015" not in har.getHeaderNames())

if __name__ == "__main__":
    unittest.main()
