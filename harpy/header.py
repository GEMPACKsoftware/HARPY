"""
Created on Mar 02 11:39:45 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

class HeaderArrayObj(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)