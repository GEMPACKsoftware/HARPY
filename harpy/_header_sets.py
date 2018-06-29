"""
Created on Jun 29 14:46:48 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

class _HeaderSet():
    """
    This class is used to represent sets associated with header arrays.
    """

    _valid_status = ["u", "e", "k"]
    _valid_dim_type = ["Set", "El", "Num"]

    def __init__(self, *args, name: str=None,
                 status: str = None,
                 dim_type: str = None,
                 dim_desc: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.status = status
        self.dim_type = dim_type
        self.dim_desc = dim_desc

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, obj):
        try:
            assert(obj in _HeaderSet._valid_status)
        except AssertionError:
            msg = "'status' must be one of %s." % ','.join(_HeaderSet._valid_status)
            raise TypeError(msg)
        self._status = obj

    @property
    def dim_type(self):
        return self._dim_type

    @dim_type.setter
    def dim_type(self, obj):
        try:
            assert(obj in _HeaderSet._valid_dim_type)
        except AssertionError:
            msg = "'dim_type' must be one of %s." % ','.join(_HeaderSet._valid_dim_type)
            raise TypeError(msg)
        self._dim_type = obj