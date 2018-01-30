import numpy
import glob
import h5py

nBatch = 10000

def nEvents_total(train):
  if train:
    files = glob.glob("prep/features_train_*.hdf5")
  else:
    files = glob.glob("prep/features_test_*.hdf5")

  nEvents = 0
  for file in files:
    f = h5py.File(file, "r")
    nEvents += len(f['label'])

    f.close()
  return nEvents 

def generate(train, nEvents, use_global = True, use_ptRel = True): # opts should be a numpy array of strings
  while 1:
    print('Beginning of data')
    if train:
      files = glob.glob("prep/features_train_*.hdf5")
    else:
      files = glob.glob("prep/features_test_*.hdf5")
    nEventsDelivered = 0
    finished = False
    for file in files:
      if finished:
        break
      f = h5py.File(file, "r")

      global_features = f['global']
      charged_pf_features = f['charged_pf']
      photon_pf_features = f['photon_pf']
      neutralHad_pf_features = f['neutralHad_pf']
      label = f['label']

      if not use_global:
        global_features = f['relIso']
      if not use_ptRel:
        charged_indices = numpy.array([0,1,2,4,5,6])
        photon_indices = numpy.array([0,1,2,4])
        neutralHad_indices = numpy.array([0,1,2,4])
     
        charged_pf_features = charged_pf_features[:,:,charged_indices]
        photon_pf_features = photon_pf_features[:,:,photon_indices]
        neutralHad_pf_features = neutralHad_pf_features[:,:,neutralHad_indices]

      counter = -nBatch
      while counter+nBatch < len(label):
        if finished:
          break
        counter += nBatch
        nEventsRemain = nEvents - nEventsDelivered
        nEventsLeftInFile = len(label) - counter
        if nEventsRemain > nBatch and nEventsLeftInFile > nBatch:
          nEventsDelivered += nBatch
          yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        elif nEventsRemain > nBatch and nEventsLeftInFile < nBatch:
          nEventsDelivered += nEventsLeftInFile
          yield [charged_pf_features[counter:counter+nBatch], photon_pf_features[counter:counter+nBatch], neutralHad_pf_features[counter:counter+nBatch], global_features[counter:counter+nBatch]], label[counter:counter+nBatch]
        elif nEventsRemain < nBatch and nEventsLeftInFile > nEventsRemain:
          finished = True
          nEventsDelivered += nEventsRemain
          yield [charged_pf_features[counter:counter+nEventsRemain], photon_pf_features[counter:counter+nEventsRemain], neutralHad_pf_features[counter:counter+nEventsRemain], global_features[counter:counter+nEventsRemain]], label[counter:counter+nEventsRemain]
        elif nEventsRemain < nBatch and nEventsLeftInFile < nEventsRemain:
          nEventsDelivered += nEventsLeftInFile
          yield [charged_pf_features[counter:counter+nEventsLeftInFile], photon_pf_features[counter:counter+nEventsLeftInFile], neutralHad_pf_features[counter:counter+nEventsLeftInFile], global_features[counter:counter+nEventsLeftInFile]], label[counter:counter+nEventsLeftInFile]  

def nSteps(train, nEvents):
  if train:
    files = glob.glob("prep/features_train_*.hdf5")
  else:
    files = glob.glob("prep/features_test_*.hdf5")

  nSteps = 0
  nEventsDelivered = 0
  finished = False
  for file in files:
    if finished:
      break
    f = h5py.File(file, "r")
    nEventsInFile = len(f['label'])
    if nEventsDelivered + nEventsInFile <= nEvents:
      nEventsDelivered += nEventsInFile
      nSteps += nEventsInFile//nBatch
      if nEventsInFile % nBatch != 0:
        nSteps += 1
    elif nEventsDelivered + nEventsInFile > nEvents:
      nEventsRemain = nEvents - nEventsDelivered
      nSteps += nEventsRemain//nBatch
      if nEventsRemain % nBatch != 0:
        nSteps += 1
      finished = True
  return nSteps
