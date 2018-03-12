"""
Created on Mar 02 11:39:45 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
import numpy as np


class HeaderArrayObj(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_valid(self, raise_exception=True) -> bool:
        """
        :return:
        """

        required_keys = ["array", "name", "long_name", "data_type", "version", "storage_type", "file_dims"]
        key_present = [key in self for key in required_keys]

        if not all(key_present):
            if raise_exception:
                idx = key_present.index(False)
                raise KeyError("'%s' not in HeaderArrayObj." % required_keys[idx])
            else:
                return False

        if not isinstance(self["name"], str):
            if raise_exception:
                raise TypeError("HeaderArrayObj 'name' must be of type 'str'.")
            else:
                return False

        if not isinstance(self["array"], np.ndarray):
            if raise_exception:
                raise TypeError("HeaderArrayObj 'array' must be of type 'numpy.ndarray'.")
            else:
                return False

        return True