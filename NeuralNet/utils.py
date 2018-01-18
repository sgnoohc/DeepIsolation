import numpy
import math

def padArray_v1(array):
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

def padArray(array, maxCands):
  nData = len(array[0])
  nFeatures = len(array)
  
  y = numpy.zeros((nData, maxCands, nFeatures))
  for i in range(nData):
    for j in range(nFeatures):
      for k in range(len(array[j][i])):
        y[i][k][j] = array[j][i][k]

  return y

def preprocess(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array 
  mean = numpy.mean(array)
  std = numpy.std(array)
  array += -mean
  array *= 1/std
  return array

def preprocess_pf(array, idx):
  tempArray = []
  for element in array:
    for subElement in element:
      tempArray.append(subElement)
  std = numpy.std(tempArray)
  array *= 1/std
  mean = numpy.mean(tempArray)
  array += -mean
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
