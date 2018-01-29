"""
Tests the read HAR functionality of HARPY. Class has been written for Python 3, and no effort has been made to make \
this class Python 2 compatible.

Created on Jan 29 11:57:37 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import sys
import os
import unittest

sys.path.append(os.path.relpath('../..'))

import HARPY

class TestReadHeader(unittest.TestCase):

    def test_readheader(self):
        har = HARPY.HAR("testdata/Mdatnew7.har", "r")
        h = har.getHeader("MAR1") # Historically, HARPY has thrown exception at this point for Python 3

if __name__ == "__main__":
    unittest.main()
