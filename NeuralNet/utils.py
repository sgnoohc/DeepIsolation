import numpy
import math
from sklearn.preprocessing import quantile_transform

def padArray(array, candLimit):
  lengths = [len(X) for X in array[0]]
  maxCands = min(candLimit, max(lengths))
  nData = len(array[0])
  nFeatures = len(array)

  y = numpy.ones((nData, maxCands, nFeatures))
  y *= -999
  for i in range(nData):
    for j in range(nFeatures):
      for k in range(min(maxCands, len(array[j][i]))):
        y[i][k][j] = array[j][i][k]

  return y

def preprocess(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array 
  array = array.astype(float)
  mean = numpy.mean(array)
  std = numpy.std(array)
  array += -mean
  array *= 1/std
  return array

def preprocess_sigmoid(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array
  array = array.astype(float)
  mean = numpy.mean(array)
  std = numpy.std(array)
  alpha = (array - mean)*(1/std)
  array = (1 - numpy.exp(-alpha))/(1 + numpy.exp(-alpha))
  return array

def preprocess_alt(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array

  maxVal = max(array)
  minVal = min(array)

  array = array - minVal
  array *= 1/(maxVal - minVal)
  return array

def preprocess_pf(array, candLimit):
  tempArray = []
  for element in array:
    for i in range(min(len(element), candLimit)):
      tempArray.append(element[i])
  std = numpy.std(tempArray)
  mean = numpy.mean(tempArray)
  array += -mean
  array *= 1/std
  return array

def preprocess_pf_sigmoid(array, candLimit):
  tempArray = []
  for element in array:
    for i in range(min(len(element), candLimit)):
      tempArray.append(element[i])
  std = numpy.std(tempArray)
  mean = numpy.mean(tempArray)
  alpha = (tempArray - mean)*(1/std)
  tempArray = (1 - numpy.exp(-alpha))/(1 + numpy.exp(-alpha))

  print('Sigmoid transformation: max, min, mean, element 1, element 2, element 3')
  print(max(tempArray))
  print(min(tempArray))
  print(numpy.mean(tempArray))
  print(tempArray[0])
  print(tempArray[1])
  print(tempArray[2])
  
  idx = 0
  for i in range(len(array)):
    for j in range(min(len(array[i]), candLimit)):
      array[i][j] = tempArray[idx]
      idx += 1
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
