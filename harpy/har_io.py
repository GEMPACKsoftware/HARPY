"""
Created on Mar 02 10:23:40 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
from collections import OrderedDict
from typing import List, Union
import io
import struct
import sys

import numpy as np

import harpy.header as header
from .HeaderCommonIO import readHeader1C, readHeader7D, readHeader2D, read7DArray, read2DArray

# compatibility function for python 2.7/3.x
if sys.version_info < (3,):
    as_unicode = False
    def tb(x):
        return x

    def fb(x):
        return x
else:
    import codecs
    as_unicode = True

    def tb(x):
        if not x: return x
        try:
            return codecs.latin_1_encode(x)[0]
        except:
            return x

    def fb(x):
        # type: str -> str
        if not x: return x
        return x.decode('utf-8')

V1DataTypes = ['1C', '2R', '2I', 'RE', 'RL', 'DE', 'DL']
StorageTypes = ['FULL', 'SPSE']
MaxDimVersion = [0, 7, 0, 14]


class HarFileInfo(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HAR_IO(object):

    def __init__(self, filename):
        # type: (str) -> HAR_IO
        """
        :arg (str) fname: Name of file to open.
        :arg (char) mode: 'r': read, 'w': write or 'a': append.
        :rtype: HAR_IO
        """

        # self.endian = "="
        # self.header = "i"
        # self._HeaderDict = OrderedDict()
        self._HeaderPos = OrderedDict()
        self.readHeaderNames(filename)

    @staticmethod
    def readHeaderNames(filename: str):
        return list(HAR_IO.readHarFileInfo(filename)["headers"].keys())

    @staticmethod
    def readHarFileInfo(filename: str):
        """
        :param filename: Filename.
        :return: An `collections.OrderedDict` object, with keys as header names.
        """
        hfi = HarFileInfo()
        hfi["file"] = filename
        hfi["headers"] = OrderedDict()

        with open(filename, "rb") as f:
            f.seek(0)
            while True:
                # Read all header names
                pos, name, end_pos = HAR_IO._readHeaderPosName(f)
                if not name:
                    break
                hfi["headers"][name] = {"pos_name": pos, "pos_data": end_pos}
        return hfi

    @staticmethod
    def _readHeaderPosName(fp: io.BufferedReader):
        """
        Identifies position of header name and name itself.

        :param file fp: a file pointer
        :return:
        """
        # type: () -> (int, str)
        pos = fp.tell()
        data = ''
        while True:
            Hpos = pos
            nbyte = HAR_IO._getEntrySize(fp)
            if not nbyte: break
            data = fp.read(4).strip()
            if data:
                data += fp.read(nbyte - 4)
                HAR_IO.checkRead(fp, nbyte)
                break
            data = None
            pos = pos + 8 + nbyte
            fp.seek(pos - 4)
            HAR_IO.checkRead(fp, nbyte)
        return Hpos, fb(data), fp.tell()

    @staticmethod
    def _getEntrySize(fp: io.BufferedReader) -> int:
        """
        Reads 4 bytes (corresponds to the size of the entry).
        :param fp: Read directly from ``fp``.
        """
        data = fp.read(4)
        if not data:
            return None
        tmp = struct.unpack("=i", data)[0]
        # print("har_io struct ", tmp)
        return tmp

    @staticmethod
    def checkRead(fp, nbyte):
        if HAR_IO._getEntrySize(fp) != nbyte:  # Must be int at end that says how long the entry was as well...
            import traceback
            traceback.print_stack()
            raise IOError('File Corrupted, start int does not match end int ')

    @staticmethod
    def readHeader(hfi: 'HarFileInfo', header_name: str, *args, **kwargs):
        """

        :param hfi: HarFileInfo object with file information.
        :param header_name: The name of the header.
        :return: Header object with data.
        """
        header_dict = hfi["headers"][header_name]

        with open(hfi["file"], "rb") as fp:

            try:
                fp.seek(header_dict["pos_name"])
            except KeyError:
                raise KeyError("Header '%s' does not exist in file." % header_name)

            fp.seek(header_dict["pos_data"])
            ret = HAR_IO.parseSecondRec(fp, header_name)
            header_dict.update(ret)

            # # readHeader methods alter self._DataObj, self.RealDim, self.DataDimension, self.StorageType possibly self.f
            if header_dict["version"] == 1:
                header_dict["header_type"] = "data"
                print("har_io header_dict ", header_dict)
                if header_dict["data_type"] == '1C':
                    header_dict["array"] = HAR_IO._read1CArray(fp, header_dict)
                    print("har_io header_dict ", header_dict)
                    if header_dict["long_name"].lower().startswith('set '):
                        header_dict["header_type"] = "set"
                        header_dict["_setNames"] = [header_dict["long_name"].split()[1]]
                elif DataType == 'RE':
                    self.hasElements = True
                    readHeader7D(self, True)
                # elif DataType == 'RL':
                #     readHeader7D(self, False)
                # elif DataType == '2R':
                #     readHeader2D(self, 'f')
                # elif DataType == '2I':
                #     readHeader2D(self, 'i')

            assert isinstance(header_dict, dict)
            assert isinstance(header_dict, dict)


    @staticmethod
    def parseSecondRec(fp: io.BufferedReader, name):

        secondRecordForm = "=4s2s4s70si"

        nbyte = HAR_IO._getEntrySize(fp)

        Record = HAR_IO._unpack_data(fp, secondRecordForm)

        print("har_io.parseSecondRec() Record ", Record)

        if fb(Record[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions of 2nd Record.")

        if fb(Record[1]) in V1DataTypes:
            Version = 1
            DataType = fb(Record[1])
        else:
            Version = int(Record[1])

            if Version > 3:
                raise RuntimeError('Header "' + name + '" is HAR Version ' + fb(Record[1]) +
                                " format which cannot not be read.\nPlease check for updates of The HARpy module")

        Storage = fb(Record[2])
        if not Storage in StorageTypes:
            raise ValueError('Unknown Storage Type "' + Storage + '" in Header "' + name + '"')

        LongName = fb(Record[3])

        Rank = Record[4]
        if Record[4] > MaxDimVersion[Version]:
            raise TypeError("Array Rank " + str(Record[4]) + '  in header "' + name + \
                            '" exceeds maximum permitted dimension for HAR Version' + str(Version))
        if 84 + 4 * Record[4] != nbyte:
            raise ValueError('Header "' + name + '" is corrupted at dimensions in second Record')

        print("har_io.parseSecondRec fp.tell() ", fp.tell())
        Sizes = HAR_IO._unpack_data(fp, "=" + ("i" * Record[-1]))

        HAR_IO.checkRead(fp, nbyte)
        return {"version": Version, "data_type": DataType, "storage_type": Storage,
                "long_name": LongName, "file_dim": Sizes}

    @staticmethod
    def _unpack_data(fp, form, data=''):
        if not data:
            data = fp.read(struct.calcsize(form))
        return struct.unpack(form, data[0:struct.calcsize(form)])

    @staticmethod
    def _read1CArray(fp, hd: dict, parent_HAR_IO):
        hd["real_dim"] = 1
        array = np.chararray((hd["file_dims"][0],),
                             itemsize=hd["file_dims"][1],
                             unicode=as_unicode)
        HAR_IO.readCharVec(fp, array)

        # DataDimension = [parent_HAR_IO.FileDims[0]]
        array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def readCharVec(fp: io.BufferedReader, array: np.ndarray):
        """
        Modifies ``array`` inplace.

        :param fp:
        :param array:
        :return:
        """
        Clen = array.itemsize

        if "<U" in str(array.dtype):
            Clen = Clen // 4

        MaxEntry = array.size
        NRec = 100
        ndata = 0
        dataFormStart = '=4siii'
        while NRec > 1:
            nbyte = HAR_IO._getEntrySize(fp)

            V = HAR_IO._unpack_data(fp, dataFormStart)

            if fb(V[0]) != '    ':
                return "Encountered characters at first four positions"

            NRec = V[1]
            if V[2] != MaxEntry:
                return "Different Size than specified"

            if ndata + V[3] > MaxEntry:
                return "More data on Header than declared"

            AllStr = fb(fp.read(V[3] * Clen))

            if nbyte != HAR_IO._getEntrySize(fp):
                return "I/O Error: sizes on 1C header do not match record length"

            for j, i in enumerate(range(0, V[3] * Clen, Clen)):
                array[ndata + j] = AllStr[i:i + Clen]

            ndata += V[3]
        return None
