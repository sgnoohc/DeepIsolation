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

def padArray_v2(array, candLimit):
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

  return y, maxCands

def padArray_v3(array):
  lengths = [len(X) for X in array[0]]
  maxCands = max(lengths)
  nData = len(array[0])
  nFeatures = len(array)

  y = numpy.ones((nData, maxCands, nFeatures))
  y *= -999
  for i in range(nData):
    for j in range(nFeatures):
      for k in range(len(array[j][i])):
        y[i][k][j] = array[j][i][k]

  return y, maxCands



def padArray_cdf(array):
  lengths = [len(X) for X in array[0]]
  maxCands = max(lengths)
  nData = len(array[0])
  nFeatures = len(array)

  y = numpy.zeros((nData, maxCands, nFeatures))
  for i in range(nData):
    for j in range(nFeatures):
      for k in range(len(array[j][i])):
        y[i][k][j] = array[j][i][k] # data, cands, features
  
  y = y.transpose((2,0,1)) # features, data, cands

  print(len(y))
  for i in range(len(y)):
    print('quantile transforming feature')
    quantile_transform(y[i])
  y = y.transpose((1,2,0)) # data, cands, features
  return y, maxCands

def preprocess(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array 
  array = array.astype(float)
  mean = numpy.mean(array)
  std = numpy.std(array)
  array += -mean
  array *= 1/std
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
  array *= 1/std
  mean = numpy.mean(tempArray)
  array += -mean
  return array

def preprocess_pf_alt(array):
  tempArray = []
  for element in array:
    for subElement in element:
      tempArray.append(subElement)
  maxVal = max(tempArray)
  minVal = min(tempArray)

  array = array - minVal
  array *= 1/(maxVal - minVal)
  return array

def preprocess_cdf(array):
  if onlyZerosAndOnes(array): # don't preprocess array if it contains only 0's and 1's (e.g. lepton_flavor).
    return array

  return quantile_transform(array, axis=1)
    
def preprocess_pf_cdf(array):
  for element in array:
    quantile_transform(element, axis=1)
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
