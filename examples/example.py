from __future__ import print_function, absolute_import

#ugly trick to make the from HARPY import
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from HARPY import HAR, Header
import numpy as np

# Create a HAR object for output, "w" open for writes, deletes file if previously open
outHAR=HAR('output.har','w')

# Data is stored in numpy array, currently only 32-bit data types possible
tmpArray=np.random.rand(4,4,4).astype('f')

# Set names for all dimensions
sets=['S1','S2','S3']

# for creation, element names have to be stored as a list of lists in the order of set appearance
SetElLists=[]
for set_name in sets:
    SetElLists.append( [set_name+"_"+str(i) for i in range(0,4) ] )

# create a Header object with these information
randomHeader=Header.HeaderFromData("RAND",tmpArray,"random Array","rand",sets,SetElLists)

# add it to the output file
outHAR.addHeader(randomHeader)

# print the output File
outHAR.write_HAR_File()

# # open in read mode, prevents writing
# inHar=HAR("output.har",'r')
#
# # difference between successive har files
# diff_between_Har=HAR.diffhar(["output.har","output.har"],"difference.har")
#
# # List of headers on file
# headersOnFile=inHar.HeaderNames()
#
# print (headersOnFile)
#
# # get a specific Header
# inHeader=inHar.getHeader("RAND")
#
# # subtract the Header
# HeaderDiff=inHeader-inHeader
# HeaderDiff.HeaderName="DIFF"
#
# HeaderAdd=2/inHeader*inHeader
#
# outHAR.addHeader(HeaderDiff)
#
# # add two headers
# HeaderSum=inHeader+randomHeader
# HeaderSum.HeaderName="SUM"
#
# outHAR.addHeader(HeaderSum)
#
# # combine multiple headers in new dimension
# HeaderCon=Header.concatenate([HeaderSum,randomHeader])
# HeaderSum.HeaderName="Cmb"
# outHAR.addHeader(HeaderCon)
#
# # get Part of Header
# print (randomHeader.DataObj[0:3:2,1:3,[-1]])
# subHeader=randomHeader[ "S1_0":"S1_3":2 ,1:3, [-1]]
# HeaderSum.HeaderName="Sub"
# #subHeader.SetNames=["NEW0","NEW1"]
# print (subHeader)
# myList=subHeader.toIndexList()
# for index in myList:
#     print(index)
#
#
# outHAR.addHeader(subHeader)
# outHAR.write_HAR_File()
#
# inHar2=HAR("output.har",'r')
#
#
