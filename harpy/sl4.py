from __future__ import print_function, absolute_import
from .har_file import HarFileObj
from .header_array import HeaderArrayObj
import numpy as np


class SL4(object):

    def __init__(self, fname):
        self._sets=[]
        self._variables=[]
        self.setHeaders={}
        self.variableDict={}
        self.varTypeDict={}
        self.solFile=HarFileObj("fname"+".sol")
        self.decode_SL4(fname)
        

    @property
    def variableNames(self):
        return self._variables
    @variableNames.setter
    def variableNames(self, val):
        raise Exception ("Variables in SL4 class cannot be assigned")

    @property
    def setNames(self):
        return self._sets
    @setNames.setter
    def setNames(self,val):
        raise Exception ("Sets in SL4 class cannot be assigned")

    def varType(self,name):
        if not name.strip().lower() in self.varTypeDict:
            raise Exception("Could not find variable '" + name + "' in the SL4 file. Please check the variable list")
        return self.varTypeDict[name.strip().lower()]

    def getSet(self,name):
        if not name.strip().lower() in self.setHeaders:
            raise Exception ("Could not find set '"+name+"' in the SL4 file. Please check the set list")
        return self.setHeaders[name.strip().lower()]

    def getVariable(self,name):
        if not name.strip().lower() in self.variableDict:
            raise Exception("Could not find variable '" + name + "' in the SL4 file. Please check the variable list")
        return self.variableDict[name.strip().lower()]

    def decode_SL4(self, fname):
        HarObj=HarFileObj(fname)


        #collect the set information. Not sure whether intertemporals work yet
        setNames =   HarObj["STNM"]
        self.generateSetHeaders(HarObj, setNames)
        resultsSet=["Cumulative"]
        resultsDataHeaders=["CUMS"]
        resultsShockComponents = ["SHCK"]
        resultsShockList = ["SHCL"]
        # check for subtotals
        if "STLS" in HarObj.getHeaderArrayNames():
            self.appendSubtotals(HarObj, resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList)


        # Get the common header needed to extract the variable
        varNames=   HarObj["VARS"]
        varDims =   HarObj["VCNA"]
        varSetPtr = HarObj["VCAR"]
        varLabel  = HarObj["VCLB"]
        varSizeEnd = HarObj["ORND"]
        varSizeExo = HarObj["OREX"]
        varExoList = HarObj["OREL"]
        cumResCom = HarObj["CMND"]
        cumResPtr = HarObj["PCUM"]
        shockPtr  = HarObj["PSHK"]
        shockVal  = HarObj["SHOC"]


        varLower=[name.strip().lower() for name in varNames.array]
        self.varTypeDict=dict(zip(varLower,HarObj["VCTP"].array))
        #prepare the different results. By default add cumulative. If subtatoals are present append them to the lists



        self._variables=[name.strip() for name in varNames.array.tolist()]
        varSetDict={}
        nvar=len(self._variables)
        nexoUsed=0; setPos=0
        self.variableDict={}

        #extract the variables. The resulting headers do not distinguish between endo and exo
        for i in range(0,nvar):
            setPos=self.generateSetDictEntry(i, resultsSet, setNames, setPos, varDims, varSetDict, varSetPtr)

            nendo, nexo, outDataList = self.assembleVariableData(HarObj, cumResCom, cumResPtr, i, nexoUsed,
                                                                 resultsDataHeaders, resultsShockComponents,
                                                                 resultsShockList, shockPtr, shockVal, varExoList,
                                                                 varSizeEnd, varSizeExo)

            self.reshapeAndAdd(i, outDataList, varLabel, varSetDict)
            if nexo != 0 and nendo != 0: nexoUsed += nexo


    def generateSetHeaders(self, HarObj, setNames):
        setSizes = HarObj["SSZ"]
        setLabels = HarObj["STLB"]
        setElStat = HarObj["ELST"]
        setEls = HarObj["STEL"]
        setElPtr = HarObj["ELAD"]
        self._sets = [name.strip().lower() for name in setNames.array.tolist()]
        nsets = len(self._sets)
        self.setHeaders = {}
        for i in range(0, nsets):
            if setElStat.array[i] == 'k':
                start = (setElPtr.array[i, 0] - 1)
                end = start + setSizes.array[i, 0]
                self.setHeaders[self._sets[i].strip().lower()] = HeaderArrayObj.SetHeaderFromData(self._sets[i],
                                                                                                  setEls.array[
                                                                                                  start:end],
                                                                                                  setLabels.array[i])

    def reshapeAndAdd(self, i, outDataList, varLabel, varSetDict):
        flatData = np.concatenate(outDataList)
        varSets = [thisSet.strip() for thisSet in varSetDict[self._variables[i].strip().lower()]]
        simSizes = tuple(
            [self.getSet(thisSet).array.shape[0] for thisSet in varSetDict[self._variables[i].strip().lower()]])
        setElDict = {}
        for myset in varSets:
            setElDict[myset.strip().lower()] = self.getSet(myset).array.tolist()
        finalData = flatData.reshape(simSizes,order="F").astype(np.float32)
        # create headers for all variables
        self.variableDict[self._variables[i].strip().lower()] = \
            HeaderArrayObj.HeaderArrayFromData(finalData,self._variables[i].strip()[0:12],varLabel.array[i][0:70], varSets, setElDict)

    def assembleVariableData(self, HarObj, cumResCom, cumResPtr, i, nexoUsed, resultsDataHeaders,
                             resultsShockComponents, resultsShockList, shockPtr, shockVal, varExoList, varSizeEnd,
                             varSizeExo):
        nexo = varSizeExo.array[i, 0]
        nendo = varSizeEnd.array[i, 0]
        # Assemble the data into a vector (subtotals are appended to the list as they are in the results* Lists)
        outDataList = []
        for DataHead, ShockComHead, ShockListHead in zip(resultsDataHeaders, resultsShockComponents, resultsShockList):
            cumRes = HarObj[DataHead]
            shockCom = HarObj[ShockComHead]
            shockList = HarObj[ShockListHead]

            start = cumResPtr.array[i, 0] - 1
            end = start + cumResCom.array[i, 0]
            Data = np.asfortranarray(cumRes.array[start:end, 0])

            nshk = shockCom.array[i, 0]
            if nexo != 0 and nendo != 0:
                insertMask = []
                for j in range(nexoUsed, nexoUsed + nexo):
                    insertMask.append(varExoList.array[j, 0] - (j - nexoUsed + 1))

                flatData = np.insert(Data, insertMask, 0)

                self.insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal)
            elif nendo != 0:
                flatData = Data
            else:
                flatData = np.zeros(nexo)
                self.insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal)
            outDataList.append(flatData)
        return nendo, nexo, outDataList

    def generateSetDictEntry(self, i, resultsSet, setNames, setPos, varDims, varSetDict, varSetPtr):
        ndim = varDims.array[i, 0]
        if ndim > 0:
            varSetDict[self._variables[i].strip().lower()] = setNames.array[[j - 1 for j in varSetPtr.array[setPos:setPos + ndim, 0]]].tolist()
            varSetDict[self._variables[i].strip().lower()] = [name.strip() for name in varSetDict[self._variables[i].strip().lower()]]
            setPos += ndim
        else:
            varSetDict[self._variables[i].strip().lower()] = []
        if len(resultsSet) > 1: varSetDict[self._variables[i].strip().lower()].append("#RESULTS")
        return setPos

    def appendSubtotals(self, HarObj, resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList):
        nresults = HarObj["STLS"].array.flatten()[0]
        for i in range(1, nresults + 1):
            resultsDataHeaders.append("%03iS" % i)
            resultsShockComponents.append("%03iC" % i)
            resultsShockList.append("%03iL" % i)
            #resultsSet.append("Subtotal%03i" % i)
        description = ["Cumulative Results"]
        description.extend(HarObj["STDS"].array.tolist())
        for name in HarObj["STDS"].array:
            resultsSet.append(str(name) if len(name) <= 12 else str(name)[0:12])
        self._sets.append("#results")
        self.setHeaders["#results"] = HeaderArrayObj.SetHeaderFromData("#RESULTS", np.array(resultsSet), "Cumlative and Subtotal elements")

    @staticmethod
    def insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal):
        if nshk > 0:
            start = shockPtr.array[i, 0] - 1
            for j in range(0, nshk):
                shkInd = start + j
                if nshk == nexo:
                    varInd=j
                else:
                    varInd = shockList.array[j, 0] - 1
                flatData[varInd] = shockVal.array[shkInd, 0]

