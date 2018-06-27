from __future__ import print_function
import struct
import sys
import os
import numpy as np
import math

__docformat__ = 'restructuredtext en'

# compatibility function for python 2.7/3.x
if sys.version_info < (3,):
    def tb(x):
        return x


    def fb(x):
        return x
else:
    import codecs


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


class HAR_IO(object):
    _HeaderPos = {}
    _HeaderDict = {}

    def __init__(self, fname, mode):
        # type: (str) -> HAR_IO
        """
        :arg (str) fname: Name of file to open.
        :arg (char) mode: 'r': read, 'w': write or 'a': append.
        :rtype: HAR_IO
        """
        if mode not in "arw": raise Exception("Unknown mode to open file")
        if mode == "r" and not os.path.isfile(fname):
            raise Exception("File "+fname+" does not exist")

        self.fname = fname
        self.mode  = mode
        self.endian = "="
        self.header = "i"


    def __enter__(self):

        self.f = open(self.fname, self.mode + '+b')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def nextHeader(self):
        # type: () -> (int, str)
        pos = self.f.tell()
        data = '';
        Hpos = 0
        while True:
            Hpos = pos
            nbyte = self.getEntrySize()
            if not nbyte: break
            data = self.f.read(4)
            if data.strip():
                data += self.f.read(nbyte - 4)
                self.checkRead(nbyte)
                break
            data = None
            pos = pos + 8 + nbyte
            self.f.seek(pos - 4)
            self.checkRead(nbyte)
        return Hpos, fb(data)

    def writeHeaderName(self, name):
        if len(name) > 4: raise Exception('Header Name ' + name + ' too long. Not written to file')
        dataForm = '=i4si'
        self.f.write(struct.pack(dataForm, 4, tb(name), 4))

    def parseSecondRec(self, name):

        secondRecordForm = self.endian + "4s2s4s70si"
        nbyte = self.getEntrySize()

        if not nbyte:
            raise Exception('Header "' + name + '" is last entry on file and does not contain additional information')

        Record = self.unpack_data(secondRecordForm, 'Second record corrupted in Header "' + name + '"')

        if fb(Record[0]) != '    ': raise Exception("Encountered characters at first four positions of 2nd Record")

        if fb(Record[1]) in V1DataTypes:
            Version = 1
            DataType = fb(Record[1])
        else:
            Version = int(Record[1])

            if Version > 3:
                raise Exception('Header "' + name + '" is HAR Version ' + fb(Record[1]) +
                                " format which cannot not be read.\nPlease check for updates of The HARpy module")

        Storage = fb(Record[2])
        if not Storage in StorageTypes:
            raise Exception('Unknown Storage Type "' + Storage + '" in Header "' + name + '"')

        LongName = fb(Record[3])

        Rank = Record[4]
        if Record[4] > MaxDimVersion[Version]:
            raise Exception("Array Rank " + str(Record[4]) + '  in header "' + name + \
                            '" exceeds maximum permitted dimension for HAR Version' + str(Version))
        if 84 + 4 * Record[4] != nbyte:
            raise Exception('Header "' + name + '" is corrupted at dimensions in second Record')

        Sizes = self.unpack_data(self.endian + "i" * Record[-1],
                                 "Could not read the associated dimensions in second record of Header" + name + '"')

        self.checkRead(nbyte)
        return Version, DataType, Storage, LongName, Sizes

    def writeSecondRecord(self, List):
        nint = len(List) - 4
        if len(List[3]) != 70: raise Exception("LongName too short")
        List = [tb(x) if isinstance(x, str) else x for x in List]
        dataForm = '=i4s2s4s70s' + 'i' * nint + 'i'
        byteLen = struct.calcsize(dataForm) - 8
        List.append(byteLen)
        List.insert(0, byteLen)
        self.f.write(struct.pack(dataForm, *List))

    def readCharVec(self, array):
        Clen = array.itemsize
        if "<U" in str(array.dtype): Clen = Clen // 4
        MaxEntry = array.size
        NRec = 100
        ndata = 0
        dataFormStart = self.endian + '4siii'
        while NRec > 1:
            nbyte = self.getEntrySize()

            V = self.unpack_data(dataFormStart, "1C Header corrupted")

            if fb(V[0]) != '    ': return "Encountered characters at first four positions"
            NRec = V[1]
            if V[2] != MaxEntry: return "Different Size than specified"
            if ndata + V[3] > MaxEntry: return "More data on Header than declared"

            AllStr = fb(self.f.read(V[3] * Clen))
            if nbyte != self.getEntrySize(): return "I/O Error: sizes on 1C header do not match record length"
            for j, i in enumerate(range(0, V[3] * Clen, Clen)):
                array[ndata + j] = AllStr[i:i + Clen]
            ndata += V[3]
        return None

    def write1CData(self, array, vecDim, strLen):
        maxwrt = 29996
        maxPerLine = maxwrt // strLen
        nrec = (vecDim - 1) // maxPerLine + 1
        nLeft = vecDim
        nDone = 0
        if nrec==0:
            dataForm = '=i4siiii'
            self.f.write(struct.pack(dataForm, 16, tb('    '), 1, 0, 0,16))
        while nrec > 0:
            dataForm = '=i4siii'
            nOnRec = min(nLeft, maxPerLine)
            ndata = 16 + nOnRec * strLen
            self.f.write(struct.pack(dataForm, ndata, tb('    '), nrec, vecDim, nOnRec))
            dataForm = '=' + str(ndata - 16) + 'si'
            packStr = tb(''.join([array[i].ljust(strLen) for i in range(nDone,nDone + nOnRec)]))
            self.f.write(struct.pack(dataForm, packStr, ndata))
            nrec -= 1
            nLeft -= nOnRec;
            nDone += nOnRec

    def getSetElementInfoRecord(self):

        SetList = [];
        ElementList = [];
        SetStatus = [];

        nbyte = self.getEntrySize()

        # read the data, has to be in chunks as it is dependent on interanl size specifications
        dataForm = '=' + '4siii12si'
        V = self.unpack_data(dataForm, "SetEl Info start corrupted [1] ")

        if fb(V[0]) != '    ': raise Exception("Encountered characters at first four positions at)SetEl")

        NSets = V[3]
        Coefficient = fb(V[4])

        if NSets == 0:
            dataForm = '=i'
        else:
            dataForm = "=" + str(NSets * 12) + 's' + str(NSets) + 's' + str(NSets) + 'i' + 'i'
        V = self.unpack_data(dataForm, "SetEl Info start corrupted [2] ")
        if NSets > 0:
            SetList = [fb(V[0][i:i + 12]) for i in range(0, NSets * 12, 12)]
            SetStatus = [fb(V[1][i:i + 1]) for i in range(0, NSets)]

        Nexplicit = V[-1]
        if Nexplicit > 0:
            dataForm = '=' + str(Nexplicit * 12) + 's'
            V = self.unpack_data(dataForm, "SetEl Info start corrupted [3] ")
            ElementList = [fb(V[-1][i:i + 12]) for i in range(0, Nexplicit * 12, 12)]
        self.checkRead(nbyte)

        return Coefficient, SetList, SetStatus, ElementList

    def writeSetElInfo(self, sets, indexTypes, Elements, CName):
        tmp = {}
        elList = []
        if not sets:
            nToWrite = 0
            nSets = 0
            nElement = 0
        else:
            statusStr = '';
            outputElements = []
            for i, j, setEls in zip(sets, indexTypes, Elements):
                if j == 'Set':
                    if not i in tmp:
                        outputElements.append(setEls)
                    tmp[i] = setEls
                    statusStr += 'k'
                elif j == 'El':
                    elList.append(setEls)
                    statusStr += 'e'
                else:
                    statusStr += 'u'
            nToWrite = len(tmp)
            nElement = len(elList)
            ElementStr = tb(''.join(elList))
            statusStr = tb(statusStr)
            SetStr = tb(''.join([item.ljust(12) for item in sets]))
            nSets = len(sets)

        dataForm = '=i4siii12si'
        if nSets > 0: dataForm += str(nSets * 13) + 's' + str(nSets) + 'i'
        dataForm += 'i'
        if nElement > 0: dataForm += str(nElement * 12)+'s'
        dataForm += 'i'
        nbyte = struct.calcsize(dataForm) - 8
        writeList = [nbyte, tb('    '), nToWrite, 1, nSets, tb(CName.ljust(12)), 1]
        if nSets > 0:
            writeList.append(SetStr + statusStr)
            writeList.extend([0 for i in Elements])

        writeList.append(nElement)
        if nElement > 0: writeList.append(ElementStr)
        writeList.append(nbyte)
        self.f.write(struct.pack(dataForm, *writeList))

        if nToWrite > 0:
            for Els in outputElements:
                array = np.array(Els)
                self.write1CData(array, len(Els), 12)

    def read7DFullObj(self, array, dtype):
        nbyte = self.getEntrySize()
        dataForm = self.endian + '4sii'

        V = self.unpack_data(dataForm, "read7D Info start corrupted")

        if fb(V[0]) != '    ': raise Exception("Encountered characters at read7D[1]")
        nrec = V[1];
        NDim = V[2]
        dataForm = self.endian + 'i' * NDim

        V = self.unpack_data(dataForm, "read7D Info start [2] corrupted")
        if nbyte != self.getEntrySize(): raise Exception("Header corrupted read7D [0]")
        oldshape = array.shape
        array = array.flatten('F')
        idata = 0
        while nrec > 1:
            nbyte = self.getEntrySize()
            dataForm = '4s15i'
            V = self.unpack_data(dataForm, "read7D data[1] corrupted")
            if fb(V[0]) != '    ': raise Exception("Encountered characters at first four positions at)SetEl")
            nrec = V[1]
            if nbyte != self.getEntrySize(): raise Exception('read7D data[2] corrupted')

            nbyte = self.getEntrySize()
            ndata = (nbyte - 8) // struct.calcsize(dtype)
            dataForm = '4si' + str(ndata) + dtype
            V = self.unpack_data(dataForm, "read7D data[2] corrupted")
            if nbyte != self.getEntrySize(): raise Exception('read7D data[2])corrupted')

            if fb(V[0]) != '    ': raise Exception("Encountered characters at read7D[2]")
            nrec = V[1]
            array[idata:idata + ndata] = V[2:]
            idata += ndata
        array = array.reshape(oldshape, order='F')

        return array

    def write7DDataFull(self, array, dtype):
        StEndList = []
        for i, j in self.slice_inds(array, 7996):
            StEndList.append([i[:], j[:]])
        nrec = len(StEndList) * 2 + 1
        dataForm = '=i4sii7ii'
        nbyte = struct.calcsize(dataForm) - 8
        writeList = [nbyte, tb('    '), nrec, 7]
        writeList.extend([array.shape[i] if i < array.ndim else 1 for i in range(0, 7)])
        writeList.append(nbyte)

        self.f.write(struct.pack(dataForm, *writeList))

        array1=array.flatten('F')
        nWritten=0
        for StEnd in StEndList:
            nrec = nrec - 1
            st = StEnd[0];
            end = StEnd[1]

            PosList = [[st[i] + 1, end[i]][ind] if i < array.ndim else [1, 1][ind] for i in range(0, 7) for ind in
                       range(0, 2)]
            dataForm = '=i4s16i'
            nbyte = struct.calcsize(dataForm) - 8
            writeList = [nbyte, tb('    '), nrec]
            writeList.extend(PosList)
            writeList.append(nbyte)
            self.f.write(struct.pack(dataForm, *writeList))

            nrec = nrec - 1;
            ndata = 1
            for i, j in zip(st, end): ndata *= (j - i)
            if dtype == 'f' or dtype == 'i': nbyte = ndata * 4 + 8

            dataForm = '=i4si'
            self.f.write(struct.pack(dataForm, nbyte, tb('    '), nrec))
            dataForm = '=' + str(ndata) + dtype
            self.f.write(struct.pack(dataForm, *array1[nWritten:nWritten+ndata].flatten('F')))
            nWritten+=ndata
            dataForm = '=i'
            self.f.write(struct.pack(dataForm, nbyte))

    def read7DSparseObj(self, array, dtype):
        # type: (nparray, str) -> NONE
        nbyte = self.getEntrySize()
        dataForm = '=4siii80s'
        nrec = 50
        V = self.unpack_data(dataForm, "read7DSparse[0] Info start corrupted")
        NNonZero = V[1]
        if V[2] != 4: raise Exception("Can only read integer 4 in read7DSparse7D ")
        if V[3] != 4: raise Exception("Can only read real 4 in read7DSparse7D ")
        if nbyte != self.getEntrySize(): raise Exception('Header corrupted read7DSparse[0]')

        oldshape = array.shape

        array = array.flatten('F')

        while nrec > 1:
            nbyte = self.getEntrySize()
            dataForm = '=4siii'
            V = self.unpack_data(dataForm, "read7DSparse[1] Info start corrupted")
            if fb(V[0]) != '    ': raise Exception("Encountered characters at read7DSparse loop")
            nrec = V[1]
            NHere = V[3]
            dataForm = '=' + str(NHere) + 'i' + str(NHere) + dtype
            V = self.unpack_data(dataForm, "read7DSparse[2] Info start corrupted")
            if nbyte != self.getEntrySize(): raise Exception('Header corrupted read7DSparse)[1]')

            for i in range(0, NHere):
                array[V[i] - 1] = V[i + NHere]

        array = array.reshape(oldshape, order='F')
        return array

    def write7DSparseObj(self, array, dtype):
        NNonZero = np.count_nonzero(array)
        Comment = 80 * ' '
        dataForm = '=i4siii80si'
        self.f.write(struct.pack(dataForm, 96, tb('    '), NNonZero, 4, 4, tb(Comment), 96))
        maxData = 3996
        nrec = (NNonZero - 1) // maxData + 1
        ndata = 0;
        valList = [];
        indexList = []

        if NNonZero == 0:
            self.f.write(struct.pack('=i4siiii', 16, tb('    '), 1, 0, 0, 16))
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
                self.writeSparseList(NNonZero, dtype, indexList, ndata, nrec, valList)
                nrec = nrec - 1;
                ndata = 0

        if ndata != 0:
            indexList=indexList[0:ndata]
            valList=valList[0:ndata]
            self.writeSparseList(NNonZero, dtype, indexList, ndata, nrec, valList)

    def writeSparseList(self, NNonZero, dtype, indexList, ndata, nrec, valList):
        dataForm = '=i4siii'
        nbyte = ndata * 2 * 4 + 16
        self.f.write(struct.pack(dataForm, nbyte, tb('    '), nrec, NNonZero, ndata))
        dataForm = '=' + str(ndata) + 'i'
        self.f.write(struct.pack(dataForm, *indexList))
        dataForm = '=' + str(ndata) + dtype
        self.f.write(struct.pack(dataForm, *valList))
        self.f.write(struct.pack('=i', nbyte))

    def read2Dobject(self, array, dtype):
        # type: (numpy.array?, datatype) -> None
        nrec = 50
        arraySize = array.size
        nread = 0
        while nread != arraySize:
            nbyte = self.getEntrySize()
            dataForm = "=4siiiiiii"
            V = self.unpack_data(dataForm, "read2D [0] Info corrupted")
            if fb(V[0]) != '    ': raise Exception("Encountered characters at read2D loop")

            if V[2] != array.shape[0]: raise Exception("Mismatching row sizes on header")
            if V[3] != array.shape[1]: raise Exception("Mismatching col sizes on header")
            xsize = V[5] - V[4] + 1
            ysize = V[7] - V[6] + 1
            ndata = xsize * ysize
            nread += ndata
            dataForm = "=" + str(ndata) + dtype
            dat = self.unpack_data(dataForm, "read2D [0] Info corrupted")
            array[V[4] - 1:V[5], V[6] - 1:V[7]] = np.array(dat).reshape(xsize, ysize, order='F')
            if nbyte != self.getEntrySize(): raise Exception('Header corrupted read2D)[1]')

    def write2Dobject(self, array, dtype):
        dataForm = "=i4siiiiiii"
        maxData = 7991
        nrec = (array.size - 1) // maxData + 1
        ndata = 0
        for st, end in self.slice_inds(array, maxData):
            if array.ndim == 1:
                indexTuple = (slice(st[0], end[0]))
                ndata = (end[0] - st[0])
                nbyte = ndata * 4 + 32
                self.f.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.size, 1, st[0] + 1, end[0], 1, 1))
            elif array.ndim == 2:
                indexTuple = (slice(st[0], end[0]), slice(st[1], end[1]))
                ndata = (end[0] - st[0]) * (end[1] - st[1])
                nbyte = ndata * 4 + 32
                self.f.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.shape[0],
                                         array.shape[1], st[0] + 1, end[0], st[1] + 1, end[1]))

            dataForm1 = '=' + str(ndata) + dtype
            self.f.write(struct.pack(dataForm1, *array[indexTuple].flatten('F')))
            self.f.write(struct.pack('=i', nbyte))
            nrec = nrec - 1

    def unpack_data(self, form, error, data=''):
        try:
            if not data:
                data = self.f.read(struct.calcsize(form))
            V = struct.unpack(form, data[0:struct.calcsize(form)])
            return V
        except:
            import traceback
            traceback.print_exc()
            raise Exception(error)

    def skipRec(self, nrec):
        """
        :rtype: 
        :type nrec: int
        :param nrec: Number of records to skip

        """""

        for i in range(0, nrec):
            nbyte = self.getEntrySize()
            if not nbyte: break
            self.f.seek(self.f.tell() + nbyte)
            self.checkRead(nbyte)

    def checkRead(self, nbyte):
        if self.getEntrySize() != nbyte:
            import traceback
            traceback.print_stack()
            raise IOError('File Corrupted, start int does not match end int ')

    def getEntrySize(self):
        # type: () -> int
        data = self.f.read(4)
        if not data: return None
        return struct.unpack(self.endian + self.header, data)[0]

    def getEntry(self):
        nbyte = self.getEntrySize()
        data = self.f.read(nbyte)
        # Twofold purpose, put us at start of next entry and check that the file is sane
        if self.getEntrySize() != nbyte:
            raise IOError('File Corrupted, start int does not match end int ')
        return data

    # If at header rewinds file to beginning of header else moves forward o next entry
    def at_header(self):
        nbyte = self.getEntrySize()
        data = self.f.read(nbyte)
        if nbyte != self.getEntrySize():
            raise IOError('File Corrupted, start int does not match end int ')
        return data[0:4].strip() > 0, data

    def seek(self, pos):
        self.f.seek(pos)

    def truncate(self):
        self.f.truncate()

    def tell(self):
        return self.f.tell()

    def readHeader(self):
        line = self.getEntry()

    def slice_inds(self, a, size):
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

        start_index = [0 for i in range(0, ndim)]

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
