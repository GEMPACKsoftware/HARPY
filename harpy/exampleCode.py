from HARPY.harpy import *
import numpy as np

# =============================== Creating Headers ===============================

# The Data array: Three dimensional random array
arrayEntries=np.random.random((3,4,5))
# creating Sets for the Header
setNames = ["Set3El", "Set4El", "Set5El"]
# creating elements for each set (list of element lists)
setElements= [
    ["E31", "E32", "E33"],
    ["E41", "E42", "E43", "E44"],
    ["E51", "E52", "E53", "E54", "E55"]
]

# short name max len 12
coefficientName="SampleHeader"
# long description max len 70
longName="Illustrates use of Harpy"

# create the Data Header
Header = HeaderArrayObj.HeaderArrayFromData(arrayEntries,coefficientName,longName,setNames, dict(zip(setNames,setElements)))

# show info except numerical values
print(Header)

# ==================================== Open a new Har file =========================
Harfile=HarFileObj("output.har")

# add the Header to the Harfile (dict like object, key is the HeaderName max len 4)
Harfile["TEST"] = Header

# To put set info on create set headers and put them into the Harfile
for i,thisSet in enumerate(setNames):
    SetHeader=HeaderArrayObj.SetHeaderFromData(thisSet,setElements[i],"Describes the set")
    Harfile["S%i"%i]=SetHeader

# write the HarFile
Harfile.writeToDisk()

#===================================  Reading from file ==============================
InFile=HarFileObj("output.har")

# get a list of all headers
HeadsOnFile = InFile.getHeaderArrayNames()
# get a specific header as HeaderArrayObj
DataHead=Harfile["TEST"]
# Can get multiple at once
SetHeadList = Harfile[ [name for name in HeadsOnFile if name != "TEST"] ]

# ===================== Accessing and setting info for existing Header ===============

# get the data. Except of setElements all data can be set in the same way swapping left and right
npDataArray = DataHead.array
setNames = DataHead.setNames
setElements = DataHead.setElements
coefficientName = DataHead.coeff_name
longName = DataHead.long_name

# ==================== getting New headers by submatrix indexing =====================

SubHead1 = DataHead[0,:,:]  # returns 2D HeaderArrayObj
SubHead1 = DataHead["E31",:,:]  # identical to above. explicit names and index position can be used interchangably everywhere
SubHead2 = DataHead[0,...]  # identical to before
SubHead3 = DataHead[[0],...] # returns 3D with first Dim single element as element is given as list
SubHead4 = DataHead[[0,2],...] # 3D first dim E31 and E33
SubHead5 = DataHead[0:2:2,...] # identical to above
SubHead5 = DataHead["E31":"E33":2,...] # identical to above
SubHead5 = DataHead[ [0,2], [0,3], [0,4]] # returns 3D Header with all dims 2 Elements

# ==================== math with Heaeders ==========================================

MathHead1 = DataHead/DataHead # returns 3D with all entries 1, elementwise division
MathHead2 = DataHead/arrayEntries # can mix numpy and Header
MathHead3 = DataHead*2.0 # affetc all entries
MathHead4 = DataHead[0,...]/np.sum(arrayEntries,axis=0) # computes the share for slice 0

# lots more math try it out or have a look in the  headerArrayObj code






