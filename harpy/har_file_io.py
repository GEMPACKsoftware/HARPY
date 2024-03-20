"""
Created on Mar 02 10:23:40 2018

"""
from collections import OrderedDict
import io
import struct
import sys
import os
import math
from typing import List,Union,BinaryIO
import numpy as np

from .header_array import HeaderArrayObj
from ._header_sets import _HeaderSet, _HeaderDims

# compatibility function for python 2.7/3.x
if sys.version_info < (3,):
    is_unicode = False
    def tb(x):
        return x

    def fb(x):
        return x
else:
    import codecs
    is_unicode = True

    def tb(x):
        if not x: return x
        try:
            return codecs.latin_1_encode(x)[0]
        except:
            return x

    def fb(x):
        if not x: return x
        return x.decode('utf-8')

class HarFileInfoObj(object):

    def __init__(self, file: str=None, ha_infos: 'List[dict]'=None):
        """
        :param str file: The absolute path to the file.
        """
        self.filename = os.path.abspath(file)
        if os.path.isfile(self.filename):
            self._mtime = os.path.getmtime(self.filename)
        if ha_infos is None:
            self._ha_infos = OrderedDict()


    def updateMtime(self):
        self._mtime = os.path.getmtime(self.filename)

    def addHAInfo(self, name, pos_name, pos_data):
        name=name.strip().upper()
        self._ha_infos[name] = HarFileInfoObj._HAInfo(name, pos_name, pos_data, parent_har_file=self)


    def getHeaderArrayNames(self):
        return list(self._ha_infos.keys())

    def items(self):
        return self._ha_infos.items()

    def __contains__(self, item):
        return item in self._ha_infos

    @property
    def file(self):
        return self.filename

    @file.setter
    def file(self, obj):
        try:
            assert(issubclass(type(obj), str))
        except AssertionError:
            msg = "'obj' is not a subclass of 'str' ('obj' is of type %s)." % type(obj)
            raise TypeError(msg)
        self.filename = obj

    @property
    def ha_infos(self):
        return self._ha_infos

    @ha_infos.setter
    def ha_infos(self, obj):
        self._ha_infos = obj

    def getHeaderArrayInfo(self, ha_name: str):
        if not ha_name.strip().upper() in self._ha_infos:
            raise ValueError("'%s' does not exist in har file '%s'." % (ha_name, self.filename))
        return self._ha_infos[ha_name.strip().upper()]

    def is_valid(self, fatal=True):
        if not os.path.isfile(self.filename):
            if fatal:
                raise FileNotFoundError("HAR file "+self.filename+" does not exist")
            else:
                return True
        valid= self._mtime == os.path.getmtime(self.filename)
        self._mtime = os.path.getmtime(self.filename)
        return valid


    class _HAInfo(object):
        """HAInfo is for Header-Array specific information. Any header array written to disk must exist in a HarFile, hence the nesting of ``_HAInfo`` within ``HarFileInfoObj``."""

        def __init__(self, name, pos_name, pos_data, parent_har_file=None):
            # TODO: Perform checks on name, pos_name, pos_data
            self.name = name
            self.pos_name = pos_name
            self.pos_data = pos_data
            self.parent_har_file = parent_har_file
            self.version = 0
            self.data_type = None
            self.storage_type = None
            self.long_name = None
            self.file_dims = None
            self.sets      = None
            self.coeff_name = None



class HarFileIO(object):

    V1SupDataTypes = ['1C', '2R', '2I', 'RE', 'RL', 'DE', 'DL'] # Supported data types for Version 1
    SupStorageTypes = ['FULL', 'SPSE'] # Supported storage types
    MaxDimVersion = [0, 7, 0, 14] # Maximum dimensions for each version???

    def __init__(self):
        """
        :rtype: HarFileIO
        """

        self._HeaderPos = OrderedDict()

    @staticmethod
    def readHarFileInfo(filename: str) -> HarFileInfoObj:
        """
        :param filename: Filename.
        :return: An `dict`, with the key-value pairs:

        """
        hfiObj = HarFileInfoObj(file=filename)

        with open(filename, "rb") as f:
            f.seek(0)
            while True:
                # Read all header names
                pos, name, end_pos = HarFileIO._readHeaderPosName(f)
                if not name:
                    break
                hfiObj.addHAInfo(name, pos, end_pos)
                hfi=hfiObj.ha_infos[name.strip().upper()]
                (hfi.version, hfi.data_type, hfi.storage_type,  hfi.long_name, hfi.file_dims) = HarFileIO._getHeaderInfo(f, name)
        return hfiObj

    @staticmethod
    def _readHeaderPosName(fp: BinaryIO):
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
            nbyte = HarFileIO._getEntrySize(fp)
            if not nbyte: break
            data = fp.read(4).strip()
            if data:
                data += fp.read(nbyte - 4)
                HarFileIO._checkRead(fp, nbyte)
                break
            data = None
            pos = pos + 8 + nbyte
            fp.seek(pos - 4)
            HarFileIO._checkRead(fp, nbyte)
        return Hpos, fb(data), fp.tell()

    @staticmethod
    def readHeaderArraysFromFile(hfi: HarFileInfoObj, ha_names: 'Union[None, str, List[str]]' = None, readData=False):

        if ha_names is None:
            ha_names = hfi.getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        haos = []

        for ha_name in ha_names or readData:
            haos.append(HarFileIO.readHeader(hfi=hfi, header_name=ha_name))

        return ha_names, haos


    @staticmethod
    def readHeader(hfi: 'HarFileInfoObj', header_name: str):
        """

        :param hfi: HarFileMemObj with file information.
        :param header_name: The name of the 
        :return: Header object with data.
        """

        hfi.is_valid()
        ha_info = hfi.getHeaderArrayInfo(header_name)

        with open(hfi.filename, "rb") as fp:

            try:
                fp.seek(ha_info.pos_name)
            except KeyError:
                raise KeyError("Header '%s' does not exist in file." % header_name)

            fp.seek(ha_info.pos_data)

            (ha_info.version, ha_info.data_type, ha_info.storage_type,  ha_info.long_name, ha_info.file_dims) = HarFileIO._getHeaderInfo(fp, header_name)

            # # readHeader methods alter self._DataObj, self.RealDim, self.DataDimension, self.StorageType possibly self.f
            if ha_info.version == 1:
                ha_info.header_type = "data"
                if ha_info.data_type == '1C':
                    ha_info.array = HarFileIO._read1CArray(fp, file_dims=ha_info.file_dims)
                    if ha_info.long_name.startswith("Set "):
                        setList = [_HeaderSet(name=ha_info.long_name.split()[1], status='k',
                                              dim_desc=ha_info.array.tolist(), dim_size=ha_info.file_dims[0])]
                    else:
                        setList = [_HeaderSet(name=None, status='n', dim_desc=None, dim_size=ha_info.file_dims[idim]) for idim in range(0, 1)]
                    ha_info.sets = _HeaderDims(setList)
                elif ha_info.data_type == 'RE':
                    ha_info.has_elements = True
                    ha_info.array = HarFileIO._readREArray(fp, ha_info, file_dims=ha_info.file_dims)
                elif ha_info.data_type == 'RL':
                    ha_info.has_elements = False
                    ha_info.array = HarFileIO._readREArray(fp, ha_info, file_dims=ha_info.file_dims,hasSets=False)
                elif ha_info.data_type in ['2R', '2I']:
                    if ha_info.data_type in ['2R']:
                        data_type = 'f'
                    else:
                        data_type = 'i'

                    setList = [_HeaderSet(name=None, status='n', dim_desc=None, dim_size=ha_info.file_dims[idim]) for idim in range(0, 2)]
                    ha_info.sets = _HeaderDims(setList)
                    ha_info.array = HarFileIO._read2DArray(fp, data_type=data_type,
                                                               file_dims=ha_info.file_dims,
                                                               storage_type=ha_info.storage_type)

                else:
                    raise ValueError("Data type '%s' is unsupported." % ha_info.data_type)
            else:
                raise RuntimeError("Unsupported/unrecognised HAR header version.")

        return HeaderArrayObj.HeaderArrayFromCompiledData(coeff_name=ha_info.coeff_name,
                                     long_name=ha_info.long_name,
                                     array=ha_info.array,
                                     SetDims=ha_info.sets)

    @staticmethod
    def _read1CArray(fp, file_dims=None, use_unicode: bool = is_unicode, ):
        array = HarFileIO._readCharVec(fp,
                                       itemsize=file_dims[1],
                                       dtype="<U12",
                                       size=(file_dims[0],),
                                       use_unicode=use_unicode)

        return np.ascontiguousarray(array)

    @staticmethod
    def _read2DArray(fp, data_type: str = "i", storage_type: str = None, file_dims: tuple = None):
        if storage_type == 'SPSE':
            raise TypeError('Sparse storage not allowed for 2D data form.')

        if data_type in ["i", "I"]:
            data_type = np.int32
            data_type_str = "i"
        elif data_type in ["f", "F"]:
            data_type = np.float32
            data_type_str = "f"
        else:
            raise ValueError(
                "Provided argument 'data_type' must be 'i' (integer) or 'f' (float). '%s' was provided." % data_type)

        # Note that 'order' refers to Fortran vs C (i.e. has nothing to do with floats or ints)
        array = np.ndarray(shape=file_dims[0:2], dtype=data_type, order='F')

        arraySize = array.size
        nread = 0
        while nread != arraySize:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = "=4siiiiiii"
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at read2D loop")
            if V[2] != array.shape[0]:
                raise ValueError("Mismatching row sizes on header")
            if V[3] != array.shape[1]:
                raise ValueError("Mismatching col sizes on header")

            xsize = V[5] - V[4] + 1
            ysize = V[7] - V[6] + 1
            ndata = xsize * ysize
            nread += ndata
            dataForm = "=" + str(ndata) + data_type_str
            dat = HarFileIO._unpack_data(fp, dataForm)
            array[V[4] - 1:V[5], V[6] - 1:V[7]] = np.array(dat).reshape(xsize, ysize, order='F')

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('Header corrupted.')

        # array = HAR_IO._read2DObj(fp, data_type, file_dims)
        array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def _readREArray(fp: BinaryIO, header_info: HarFileInfoObj._HAInfo, file_dims: tuple = None, hasSets=True):
        """

        :param fp:
        :param header_info:
        :param file_dims:
        :param hasSets:
        :return:
        """

        if hasSets:
            (header_info.coeff_name, header_info.sets) = HarFileIO._readSets(fp, file_dims=file_dims)
            tmpDim = header_info.sets.ndim()
        else:
            header_info.sets=_HeaderDims([_HeaderSet(name=None, status='n', dim_desc=None, dim_size= file_dims[idim]) for idim in range(0,7)])
            tmpDim = 7

        array = np.ndarray(shape=file_dims[0:tmpDim], dtype=np.float32, order='F')
        array.fill(0.0)

        if header_info.storage_type == 'FULL':
            array = HarFileIO._readREFullObj(fp, array, 'f')
        else:
            array = HarFileIO._readRESparseObj(fp, array, 'f')
        # this is needed for rank 0 objects
        myshape=array.shape
        return np.ascontiguousarray(array).reshape(myshape)


    @staticmethod
    def _getHeaderInfo(fp: BinaryIO, name):

        nbyte = HarFileIO._getEntrySize(fp)

        secondRecordForm = "=4s2s4s70si"
        Record = HarFileIO._unpack_data(fp, secondRecordForm)

        if fb(Record[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions of 2nd Record.")

        if fb(Record[1]) in HarFileIO.V1SupDataTypes:
            Version = 1
            DataType = fb(Record[1])
        else:
            raise RuntimeError('Header "' + name + '" is HAR Version ' + fb(Record[1]) +
                            " format which cannot not be read.\nPlease check for updates of The HARpy module")

        StorageType = fb(Record[2])
        if not StorageType in HarFileIO.SupStorageTypes:
            raise ValueError('Unknown StorageType "' + StorageType + '" in Header "' + name + '"')

        LongName = fb(Record[3])

        if Record[4] > HarFileIO.MaxDimVersion[Version]:
            raise TypeError("Array Rank " + str(Record[4]) + '  in header "' + name + \
                            '" exceeds maximum permitted dimension for HAR Version' + str(Version))
        if 84 + 4 * Record[4] != nbyte:
            raise ValueError('Header "' + name + '" is corrupted at dimensions in second Record')

        Sizes = HarFileIO._unpack_data(fp, "=" + ("i" * Record[-1]))

        HarFileIO._checkRead(fp, nbyte)
        return Version, DataType, StorageType, LongName, Sizes

    @staticmethod
    def _readCharVec(fp: BinaryIO, itemsize:int=None, dtype=None, size:tuple=None, use_unicode=is_unicode):
        """

        :param fp:
        :param itemsize:
        :param dtype:
        :param size:
        :param is_unicode:
        :return:
        """

        array = np.chararray(size, itemsize=itemsize, unicode=use_unicode)
        Clen = array.itemsize

        if "<U" in str(dtype):
            Clen = Clen // 4

        MaxEntry = size[0]
        NRec = 100
        ndata = 0
        dataFormStart = '=4siii'

        while NRec > 1:
            nbyte = HarFileIO._getEntrySize(fp)

            V = HarFileIO._unpack_data(fp, dataFormStart)

            if fb(V[0]) != '    ':
                raise IOError("Encountered characters at first four positions")

            NRec = V[1]
            if V[2] != MaxEntry:
                raise ValueError("Different Size than specified")

            if ndata + V[3] > MaxEntry:
                raise RuntimeError("More data on Header than declared")

            AllStr = fb(fp.read(V[3] * Clen))

            if nbyte != HarFileIO._getEntrySize(fp):
                raise IOError("I/O Error: sizes on 1C header do not match record length")

            for j, i in enumerate(range(0, V[3] * Clen, Clen)):
                array[ndata + j] = AllStr[i:i + Clen]
            ndata += V[3]

        return np.ascontiguousarray(array)

    @staticmethod
    def _readREFullObj(fp, array, dtype):

        nbyte = HarFileIO._getEntrySize(fp)
        dataForm = '=4sii'

        V = HarFileIO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise Exception("Encountered characters at read7D[1]")

        nrec = V[1]
        NDim = V[2]
        dataForm = "=" + ('i' * NDim)

        V = HarFileIO._unpack_data(fp, dataForm)

        if nbyte != HarFileIO._getEntrySize(fp):
            raise RuntimeError("Header corrupted read7D[0] @ %d" % fp.tell())

        oldshape = array.shape
        array = array.flatten('F')
        idata = 0
        while nrec > 1:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = '4s15i'
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at first four positions at)SetEl")

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('read7D data[2] corrupted')

            nbyte = HarFileIO._getEntrySize(fp)
            ndata = (nbyte - 8) // struct.calcsize(dtype)
            dataForm = '4si' + str(ndata) + dtype
            V = HarFileIO._unpack_data(fp, dataForm)

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('read7D data[2])corrupted')

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at read7D[2]")

            nrec = V[1]
            array[idata:idata + ndata] = V[2:]
            idata += ndata
        array = array.reshape(oldshape, order='F')
        return array

    @staticmethod
    def _readRESparseObj(fp:BinaryIO, array: np.ndarray, dtype):
        nbyte = HarFileIO._getEntrySize(fp)
        dataForm = '=4siii80s'
        nrec = 50
        V = HarFileIO._unpack_data(fp, dataForm)
        if V[2] != 4:
            raise ValueError("Can only read integer 4 in read7DSparse7D ")
        if V[3] != 4:
            raise ValueError("Can only read real 4 in read7DSparse7D ")
        if nbyte != HarFileIO._getEntrySize(fp):
            raise ValueError('Header corrupted read7DSparse[0]')

        oldshape = array.shape

        array = array.flatten('F')

        while nrec > 1:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = '=4siii'
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise ValueError("Encountered characters at read7DSparse loop")

            nrec = V[1]
            NHere = V[3]
            dataForm = '=' + str(NHere) + 'i' + str(NHere) + dtype
            V = HarFileIO._unpack_data(fp, dataForm)

            if nbyte != HarFileIO._getEntrySize(fp):
                raise ValueError('Header corrupted read7DSparse)[1]')

            for i in range(0, NHere):
                array[V[i] - 1] = V[i + NHere]

        array = array.reshape(oldshape, order='F')
        return array

    @staticmethod
    def _readSetElementInfoRecord(fp):

        SetNames = []
        ElementList = []
        SetStatus = []

        nbyte = HarFileIO._getEntrySize(fp)

        # read the data, has to be in chunks as it is dependent on interanl size specifications
        dataForm = '=' + '4siii12si'

        V = HarFileIO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions at SetEl")

        NSets = V[3]
        Coefficient = fb(V[4])
        setKnown=V[5]!=0
        if not setKnown:
            dataForm = '=i'
        else:
            dataForm = "=" + str(NSets * 12) + 's' + str(NSets) + 's' + str(NSets) + 'i' + 'i'

        V = HarFileIO._unpack_data(fp, dataForm)
        if setKnown:
            SetNames = [fb(V[0][i:i + 12]) for i in range(0, NSets * 12, 12)]
            SetStatus = [fb(V[1][i:i + 1]) for i in range(0, NSets)]

        Nexplicit = V[-1]
        dataForm = '=' + str(Nexplicit * 12) + 's'
        V = HarFileIO._unpack_data(fp, dataForm)

        if Nexplicit > 0:
            dataForm = '=' + str(Nexplicit * 12) + 's'
            V = HarFileIO._unpack_data(fp, dataForm)

            ElementList = [fb(V[-1][i:i + 12]) for i in range(0, NSets * 12, 12)]

        HarFileIO._checkRead(fp, nbyte)

        return Coefficient, SetNames, SetStatus, ElementList

    @staticmethod
    def _readSets(fp: BinaryIO, file_dims=None) -> (str,_HeaderDims):
        """
        :param fp: BinaryIO object.
        :return tuple: (coefficient_name, header_sets)
        """
        Coefficient, SetList, SetStatus, ElementList = HarFileIO._readSetElementInfoRecord(fp)

        set_names = [name.strip() for name in SetList]

        idim = 0
        header_sets = []
        processedSet = OrderedDict()
        for name, status in zip(set_names, SetStatus):
            if status == 'k':
                if name not in processedSet:
                    processedSet[name] = HarFileIO._readCharVec(fp, itemsize=12, use_unicode=is_unicode, size=tuple([file_dims[idim]]), dtype="<U12")
                header_sets.append(_HeaderSet(name=name, status=status, dim_desc=[item.strip() for item in processedSet[name]], dim_size= file_dims[idim]))
            elif status == 'u':
                header_sets.append(_HeaderSet(name=name, status=status, dim_desc=None, dim_size= file_dims[idim]))
            elif status == 'e':
                header_sets.append(_HeaderSet(name=name, status=status, dim_desc=ElementList.pop(0), dim_size= file_dims[idim]))
            idim += 1

        return Coefficient, _HeaderDims(header_sets)

    @staticmethod
    def _unpack_data(fp, form, data=''):
        """
        :param fp:
        :param form:
        :param data:
        :return:
        """
        # print("HarFileIO._unpack_data() form, data ", form, data)
        if not data:
            data = fp.read(struct.calcsize(form))
        return struct.unpack(form, data[0:struct.calcsize(form)])

    @staticmethod
    def _getEntrySize(fp: BinaryIO) -> Union[int,None]:
        """
        Reads 4 bytes (corresponds to the size of the entry).
        :param fp: Read directly from ``fp``.
        """
        data = fp.read(4)
        if not data:
            return None
        tmp = struct.unpack("=i", data)[0]
        return tmp

    @staticmethod
    def _checkRead(fp, nbyte):
        if HarFileIO._getEntrySize(fp) != nbyte:  # Must be int at end that says how long the entry was as well...
            import traceback
            traceback.print_stack()
            raise IOError('File Corrupted, start int does not match end int.')

    @staticmethod
    def writeHeaders(filename: 'Union[str, io.BufferedWriter]', hnames : 'List[str]',
                     head_arr_objs: 'Union[HeaderArrayObj, List[HeaderArrayObj]]'):

        if isinstance(head_arr_objs, HeaderArrayObj):
            head_arr_objs = [head_arr_objs]

        for head_arr_obj in head_arr_objs:
            if not isinstance(head_arr_obj, HeaderArrayObj):
                raise TypeError("All 'head_arr_objs' must be of HeaderArrayObj type.")
            head_arr_obj.is_valid()

        if isinstance(filename, str):
            fp = open(filename, "wb")
        else:
            raise TypeError("'filename' is invalid - must be either file object or string.")

        with fp:
            for hname, head_arr_obj in zip(hnames, head_arr_objs):
                header_type_str = str(head_arr_obj.array.dtype)
                has_sets = head_arr_obj.sets.defined()
                # HarFileIO._writeHeader(fp, head_arr_obj)

                if header_type_str in ['float32','float64'] and (head_arr_obj.array.ndim != 2 or has_sets):
                    HarFileIO._writeHeader7D(fp, hname, head_arr_obj)
                elif header_type_str in ['int32','int64', 'float32','float64' ]:
                    HarFileIO._writeHeader2D(fp, hname, head_arr_obj)
                elif '<U' in header_type_str or '|S' in header_type_str:
                    if head_arr_obj.array.ndim > 1:
                        print('"' + hname + '" can not be written as character arrays ndim>1 are not yet supported')
                        return
                    HarFileIO._writeHeader1C(fp, hname, head_arr_obj)
                else:
                    raise TypeError('Can not write data in Header: "' +
                                    hname + '" as data style '+header_type_str+' does not match any known Header type')
                fp.flush()



    @staticmethod
    def _writeHeader7D(fp: BinaryIO, hname : str, head_arr_obj: HeaderArrayObj):
        hasElements = head_arr_obj.sets.defined()
        dataFill = float(np.count_nonzero(head_arr_obj.array)) / head_arr_obj.array.size

        if dataFill > 0.4:
            head_arr_obj.storage_type = 'FULL'
        else:
            head_arr_obj.storage_type = 'SPSE'

        shape7D = [head_arr_obj.array.shape[i] if i < head_arr_obj.array.ndim else 1 for i in range(0, 7)]

        HarFileIO._writeHeaderName(fp, hname)
        HeaderType = 'RE'

        secRecList = ['    ', HeaderType, head_arr_obj.storage_type, head_arr_obj.long_name, 7]
        secRecList.extend(shape7D)
        HarFileIO._writeSecondRecord(fp, secRecList)
        HarFileIO._writeSetElInfo(fp, head_arr_obj)

        if head_arr_obj.storage_type == 'FULL':
            HarFileIO._write7DFullArray(fp, np.asfortranarray(head_arr_obj.array), 'f')
        else:
            HarFileIO._write7DSparseArray(fp, np.asfortranarray(head_arr_obj.array), 'f')

    @staticmethod
    def _writeHeader2D(fp: BinaryIO, hname : str, head_arr_obj: HeaderArrayObj):
        HarFileIO._writeHeaderName(fp, hname)
        typeString = str(head_arr_obj.array.dtype)
        shape2D = [head_arr_obj.array.shape[i] if i < head_arr_obj.array.ndim else 1 for i in range(0, 2)]
        if typeString == 'int32':
            dtype = 'i'
            secRecList = ['    ', '2I', 'FULL', head_arr_obj.long_name, 2]
        elif typeString == 'float32':
            secRecList = ['    ', '2R', 'FULL', head_arr_obj.long_name, 2]
            dtype = 'f'
        else:
            raise TypeError("Can only write 32bit float or int to 2D arrays")
        secRecList.extend(shape2D)

        HarFileIO._writeSecondRecord(fp, secRecList)
        HarFileIO._write2DArray(fp, np.asfortranarray(head_arr_obj.array), dtype)

    @staticmethod
    def _writeHeader1C(fp: BinaryIO, hname : str, head_arr_obj: HeaderArrayObj):

        HarFileIO._writeHeaderName(fp, hname)
        typeString = str(head_arr_obj.array.dtype)
        no_chars = int(typeString[2:])
        secRecList = ['    ', '1C', 'FULL', head_arr_obj.long_name, 2, head_arr_obj.array.size, no_chars]
        HarFileIO._writeSecondRecord(fp, secRecList)
        HarFileIO._write1CArray(fp, np.asfortranarray(head_arr_obj.array), head_arr_obj.array.size, no_chars)

    @staticmethod
    def _writeHeaderName(fp: BinaryIO, name: str):

        if len(name) > 4:
            raise ValueError('Header name ' + name + ' is longer than 4 characters long. Header array not written to file.')

        name=name.ljust(4)
        dataForm = '=i4si'
        packed = struct.pack(dataForm, 4, tb(name), 4)
        fp.write(packed)

    @staticmethod
    def _writeSecondRecord(fp: BinaryIO, inList):
        nint = len(inList) - 4

        if len(inList[3]) != 70:
            raise ValueError("'long_name' must be precisely 70 characters long. 'long_name' is: %s (%d characters long)." % (inList[3], len(inList[3])))

        inList = [tb(x) if isinstance(x, str) else x for x in inList]
        dataForm = '=i4s2s4s70s' + 'i' * nint + 'i' # For reading it is "=4s2s4s70si"
        byteLen = struct.calcsize(dataForm) - 8
        inList.append(byteLen)
        inList.insert(0, byteLen)
        packed = struct.pack(dataForm, *inList)
        fp.write(packed)

    @staticmethod
    def _write7DFullArray(fp, array, dtype):
        StEndList = []
        for i, j in HarFileIO._slice_inds(array, 7996):
            StEndList.append([i[:], j[:]])
        nrec = len(StEndList) * 2 + 1
        dataForm = '=i4sii7ii'
        nbyte = struct.calcsize(dataForm) - 8
        writeList = [nbyte, tb('    '), nrec, 7]
        writeList.extend([array.shape[i] if i < array.ndim else 1 for i in range(0, 7)])
        writeList.append(nbyte)

        fp.write(struct.pack(dataForm, *writeList))

        array1=array.flatten('F')
        nWritten=0
        for StEnd in StEndList:
            nrec = nrec - 1
            st = StEnd[0]
            end = StEnd[1]

            PosList = [[st[i] + 1, end[i]][ind] if i < array.ndim else [1, 1][ind] for i in range(0, 7) for ind in
                       range(0, 2)]
            dataForm = '=i4s16i'
            nbyte = struct.calcsize(dataForm) - 8
            writeList = [nbyte, tb('    '), nrec]
            writeList.extend(PosList)
            writeList.append(nbyte)
            fp.write(struct.pack(dataForm, *writeList))

            nrec = nrec - 1
            ndata = 1
            for i, j in zip(st, end): ndata *= (j - i)
            if dtype == 'f' or dtype == 'i': nbyte = ndata * 4 + 8

            dataForm = '=i4si'
            fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec))
            dataForm = '=' + str(ndata) + dtype
            fp.write(struct.pack(dataForm, *array1[nWritten:nWritten+ndata].flatten('F')))
            nWritten+=ndata
            dataForm = '=i'
            fp.write(struct.pack(dataForm, nbyte))

    @staticmethod
    def _write7DSparseArray(fp, array, dtype):
        NNonZero = np.count_nonzero(array)
        Comment = 80 * ' '
        dataForm = '=i4siii80si'
        fp.write(struct.pack(dataForm, 96, tb('    '), NNonZero, 4, 4, tb(Comment), 96))
        maxData = 3996
        nrec = (NNonZero - 1) // maxData + 1
        ndata = 0

        if NNonZero == 0:
            fp.write(struct.pack('=i4siiii', 16, tb('    '), 1, 0, 0, 16))
            return

        indexList=maxData*[None]
        valList=maxData*[None]
        tmp=array.flatten('F')
        nzind=np.nonzero(tmp)
        for i in nzind[0]:
            ndata += 1
            indexList[ndata-1]=i+1
            valList[ndata-1]=tmp[i]
            if ndata == maxData:
                HarFileIO._writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList)
                nrec = nrec - 1
                ndata = 0

        if ndata != 0:
            indexList=indexList[0:ndata]
            valList=valList[0:ndata]
            HarFileIO._writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList)

    @staticmethod
    def _write2DArray(fp, array, dtype):
        dataForm = "=i4siiiiiii"
        maxData = 7991
        nrec = (array.size - 1) // maxData + 1
        ndata = 0
        indexTuple=(None,)
        nbyte=0
        for st, end in HarFileIO._slice_inds(array, maxData):
            if array.ndim == 1:
                indexTuple = (slice(st[0], end[0]))
                ndata = (end[0] - st[0])
                nbyte = ndata * 4 + 32
                fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.size, 1, st[0] + 1, end[0], 1, 1))
            elif array.ndim == 2:
                indexTuple = (slice(st[0], end[0]), slice(st[1], end[1]))
                ndata = (end[0] - st[0]) * (end[1] - st[1])
                nbyte = ndata * 4 + 32
                fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.shape[0],
                                         array.shape[1], st[0] + 1, end[0], st[1] + 1, end[1]))

            dataForm1 = '=' + str(ndata) + dtype
            fp.write(struct.pack(dataForm1, *array[indexTuple].flatten('F')))
            fp.write(struct.pack('=i', nbyte))
            nrec = nrec - 1

    @staticmethod
    def _write1CArray(fp : BinaryIO, array : np.array, vecDim : int, strLen : int):
        maxwrt = 29996
        maxPerLine = maxwrt // strLen
        nrec = (vecDim - 1) // maxPerLine + 1
        nLeft = vecDim
        nDone = 0
        if nrec==0:
            dataForm = '=i4siiii'
            fp.write(struct.pack(dataForm, 16, tb('    '), 1, 0, 0,16))
        while nrec > 0:
            dataForm = '=i4siii'
            nOnRec = min(nLeft, maxPerLine)
            ndata = 16 + nOnRec * strLen
            fp.write(struct.pack(dataForm, ndata, tb('    '), nrec, vecDim, nOnRec))
            dataForm = '=' + str(ndata - 16) + 'si'
            packStr = tb(''.join([array[i].ljust(strLen) for i in range(nDone,nDone + nOnRec)]))
            fp.write(struct.pack(dataForm, packStr, ndata))
            nrec -= 1
            nLeft -= nOnRec
            nDone += nOnRec

    @staticmethod
    def _writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList):
        dataForm = '=i4siii'
        nbyte = ndata * 2 * 4 + 16
        fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, NNonZero, ndata))
        dataForm = '=' + str(ndata) + 'i'
        fp.write(struct.pack(dataForm, *indexList))
        dataForm = '=' + str(ndata) + dtype
        fp.write(struct.pack(dataForm, *valList))
        fp.write(struct.pack('=i', nbyte))

    @staticmethod
    def _writeSetElInfo(fp, header_arr_obj: HeaderArrayObj):

        sets = [setDim.name for setDim in header_arr_obj.sets.dims]
        indexTypes = [setDim.status for setDim in header_arr_obj.sets.dims]
        Elements = [setDim.dim_desc for setDim in header_arr_obj.sets.dims]

        CName = header_arr_obj.coeff_name
        tmp = {}
        elList = []
        if all(item is None for item in sets):
            nToWrite = 0
            nSets = len(sets)
            nElement = 0
            setsKnown= 0 # represents a fortran logical
        else:
            setsKnown=1
            statusStr = ''
            outputElements = []
            for i, j, setEls in zip(sets, indexTypes, Elements):
                if j == 'k':
                    if not i in tmp:
                        outputElements.append(setEls)
                    tmp[i] = setEls
                    statusStr += 'k'
                elif j == 'e':
                    elList.append(setEls[0])
                    statusStr += 'e'
                else:
                    statusStr += 'u'
            nToWrite = len(tmp)
            nElement = len(elList)
            ElementStr = tb(''.join(elList))
            statusStr = tb(statusStr)
            SetStr = tb(''.join([item.ljust(12) if not item is None else " "*12 for item in sets]))
            nSets = len(sets)

        dataForm = '=i4siii12si'
        if setsKnown == 1: dataForm += str(nSets * 13) + 's' + str(nSets) + 'i'
        dataForm += 'i'
        if nElement > 0: dataForm += str(nElement * 12)
        dataForm += 'i'
        nbyte = struct.calcsize(dataForm) - 8

        writeList = [nbyte, tb('    '), nToWrite, 1, nSets, tb(CName.ljust(12)), setsKnown]

        if setsKnown == 1:
            writeList.append(SetStr + statusStr)
            writeList.extend([0]*len(Elements) )

        writeList.append(nElement)
        if nElement > 0: writeList.append(ElementStr)
        writeList.append(nbyte)

        fp.write(struct.pack(dataForm, *writeList))

        if nToWrite > 0:
            for Els in outputElements:
                array = np.array(Els)
                HarFileIO._write1CArray(fp, array, len(Els), 12)

    @staticmethod
    def _slice_inds(a, size):
        if a.ndim==0:
            yield [0],[1]
            return
        stride = [i // a.dtype.itemsize for i in a.strides]
        offset = [i * j for i, j in zip(stride, a.shape)]
        ndim = len(offset)
        for inc_dim, off in enumerate(offset):
            if off > size: break
        nslice = size // stride[inc_dim]
        increment = int(size // stride[inc_dim])
        tot_iter = int(math.ceil(float(a.shape[inc_dim]) / nslice) * offset[-1] / offset[inc_dim])

        end_index = [0 if i == inc_dim else 1 for i in range(0, ndim)]
        end_index[0:inc_dim] = a.shape[0:inc_dim]

        start_index = [0] * ndim

        for i in range(0, tot_iter):
            if end_index[inc_dim] == a.shape[inc_dim]:
                start_index[inc_dim] = 0
                end_index[inc_dim] = increment
                for j in range(inc_dim + 1, ndim):
                    if end_index[j] == a.shape[j]:
                        start_index[j] = 0
                        end_index[j] = 1
                    else:
                        start_index[j] = end_index[j]
                        end_index[j] += 1
                        break
            else:
                start_index[inc_dim] = end_index[inc_dim]
                end_index[inc_dim] = min(a.shape[inc_dim], end_index[inc_dim] + increment)
            yield start_index, end_index
