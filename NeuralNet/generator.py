import numpy
import glob
import h5py

nBatch = 10000

def nEvents_train():
  files = glob.glob("prep/features_train_*.hdf5")

  nEvents = 0
  for file in files:
    f = h5py.File(file, "r")
    nEvents += len(f['label'])

    f.close()
  return nEvents 

def generate_train():
  while 1:
    files = glob.glob("prep/features_train_*.hdf5")
    for file in files:
      f = h5py.File(file, "r")

      global_features = f['global']
      charged_pf_features = f['charged_pf']
      photon_pf_features = f['photon_pf']
      neutralHad_pf_features = f['neutralHad_pf']
      label = f['label']

      counter = 0
      while counter < len(label):
        yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        counter += nBatch

def generate_test():
  while 1:
    files = glob.glob("prep/features_test_*.hdf5")
    for file in files:
      f = h5py.File(file, "r")

      global_features = f['global_features']
      charged_pf_features = f['charged_pf_features']
      photon_pf_features = f['photon_pf_features']
      neutralHad_pf_features = f['neutralHad_pf_features']
      label = f['label']

      counter = 0
      while counter < len(label):
        yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        counter += nBatch

def nSteps_train():
  files = glob.glob("prep/features_train_*.hdf5")

  nSteps = 0
  for file in files:
    f = h5py.File(file, "r")
    nEvents = len(f['label'])
    nSteps += nEvents//nBatch
    if nEvents % nBatch != 0:
      nSteps += 1
  return nSteps

def nSteps_test():
  files = glob.glob("prep/features_test_*.hdf5")

  nSteps = 0
  for file in files:
    f = h5py.File(file, "r")
    nEvents = len(f['label'])
    nSteps += nEvents//nBatch
    if nEvents % nBatch != 0:
      nSteps += 1
  return nSteps

