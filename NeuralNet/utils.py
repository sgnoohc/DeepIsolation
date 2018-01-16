import numpy
import math

def padArray(array):
  lengths = [len(X) for X in array[0]]
  maxCands = max(lengths)
  nData = len(array[0])
  nFeatures = len(array)

  y = numpy.zeros((nData, maxCands, nFeatures))
  for i in range(nData):
    for j in range(nFeatures):
      for k in range(len(array[j][i])):
        y[i][k][j] = array[j][i][k]

  return y, maxCands

def preprocess(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array 
  mean = numpy.mean(array)
  std = numpy.std(array)
  array += -mean
  array *= 1/std
  return array

def preprocess_pf(array, idx):
  if array.dtype == 'int32' or array.dtype == 'int64':
    return array
  tempArray = []
  for element in array:
    for subElement in element:
      #if idx == 0 or idx == 3: # for pT and pTRel 
      #  subElement = math.log(subElement)
      tempArray.append(subElement)
  mean = numpy.mean(tempArray)
  std = numpy.std(tempArray)
  array += -mean
  array *= 1/std
  return array

def onlyZerosAndOnes(array):
  for element in array:
    if not (element == 0 or element == 1):
      return False
  return True

def find_nearest(array,value):
    val = numpy.ones_like(array)*value
    idx = (numpy.abs(array-val)).argmin()
    return array[idx], idx
