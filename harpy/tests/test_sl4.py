
import os
import unittest

from ..sl4 import SL4


class TestSL4(unittest.TestCase):
    _dd = os.path.join(os.path.dirname(__file__), "testdata", "")

    def testSl4(self):
        sl4File=SL4(os.path.join(self._dd,"SJSUB.sl4"))
        setsOnFile=['sect', 'fac', 'num_sect']
        for setName in setsOnFile:
            self.assertIn(setName, sl4File.setNames)