"""
Created on Mar 02 11:40:02 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import os
import unittest

import harpy.header_array as header

class TestHeader(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def test_read_header_names(self):

        pass



if __name__ == "__main__":
    unittest.main()
