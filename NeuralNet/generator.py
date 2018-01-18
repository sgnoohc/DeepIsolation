import numpy
import glob

nBatch = 10000

def nEvents_train():
  files = glob.glob("prep/features_train_*.npz")

  nEvents = 0
  for file in files:
    npzfile = numpy.load(file)
    nEvents += len(npzfile['label'])
  return nEvents

def generate_train():
  while 1:
    files = glob.glob("prep/features_train_*.npz")
    for file in files:
      npzfile = numpy.load(file)

      global_features = npzfile['global_features']
      charged_pf_features = npzfile['charged_pf_features']
      photon_pf_features = npzfile['photon_pf_features']
      neutralHad_pf_features = npzfile['neutralHad_pf_features']
      label = npzfile['label']

      counter = 0
      while counter < len(label):
        yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        counter += nBatch

def generate_test():
  while 1:
    files = glob.glob("prep/features_test_*.npz")
    for file in files:
      npzfile = numpy.load(file)

      global_features = npzfile['global_features']
      charged_pf_features = npzfile['charged_pf_features']
      photon_pf_features = npzfile['photon_pf_features']
      neutralHad_pf_features = npzfile['neutralHad_pf_features']
      label = npzfile['label']

      counter = 0
      while counter < len(label):
        yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        counter += nBatch

