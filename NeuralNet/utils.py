import numpy

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

