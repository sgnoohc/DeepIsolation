import ROOT
import numpy
import root_numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import glob

validation_frac = 0.1 # fraction of files to use as validation (will not be trained on)

# Read data from ROOT file
filenames = glob.glob("/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIsolation_Babies/merged_ntuple_*.root")
filenames = numpy.array(filenames)

max_pf_charged = 0
max_pf_photon = 0
max_pf_neutralHad = 0


# Find maximum number of pf cands (will pad arrays to that length)
for i in range(len(filenames)):
  print(i)
  f = ROOT.TFile(filenames[i])
  tree = f.Get("t")

  branches = ['pf_charged_pt', 'pf_photon_pt', 'pf_neutralHad_pt', 'lepton_flavor']

  data = root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 1')

  charged_pf_timesteps = [len(X) for X in data['pf_charged_pt']]
  photon_pf_timesteps = [len(X) for X in data['pf_photon_pt']]
  neutralHad_pf_timesteps = [len(X) for X in data['pf_neutralHad_pt']] 

  if max(charged_pf_timesteps) > max_pf_charged:
    max_pf_charged = max(charged_pf_timesteps)
  if max(photon_pf_timesteps) > max_pf_photon:
    max_pf_photon = max(photon_pf_timesteps)
  if max(neutralHad_pf_timesteps) > max_pf_neutralHad:
    max_pf_neutralHad = max(neutralHad_pf_timesteps)

print(max_pf_charged)
print(max_pf_photon)
print(max_pf_neutralHad)

# Loop through root babies and save hdf5 files
for i in range(len(filenames)):
  print('Working on file %d/%d' % (i+1, len(filenames)))

  f = ROOT.TFile(filenames[i])
  tree = f.Get("t")

  branches = ['lepton_isFromW', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_relIso03EA', 'lepton_chiso', 'lepton_nhiso', 'lepton_emiso', 'lepton_ncorriso', 'lepton_dxy', 'lepton_dz', 'lepton_ip3d', 'nvtx', 'lepton_flavor', 'lepton_nChargedPf', 'lepton_nPhotonPf', 'lepton_nNeutralHadPf', 'pf_charged_pt', 'pf_charged_dR', 'pf_charged_alpha', 'pf_charged_ptRel', 'pf_charged_puppiWeight', 'pf_charged_fromPV', 'pf_charged_pvAssociationQuality', 'pf_photon_pt', 'pf_photon_dR', 'pf_photon_alpha', 'pf_photon_ptRel', 'pf_photon_puppiWeight', 'pf_neutralHad_pt', 'pf_neutralHad_dR', 'pf_neutralHad_alpha', 'pf_neutralHad_ptRel', 'pf_neutralHad_puppiWeight']
  data = root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 1')

  # Grab features
  global_features = numpy.array([data['lepton_pt'], data['lepton_eta'], data['lepton_phi'], data['lepton_relIso03EA'], data['lepton_chiso'], data['lepton_nhiso'], data['lepton_emiso'], data['lepton_ncorriso'], data['lepton_dxy'], data['lepton_dz'], data['lepton_ip3d'], data['nvtx'], data['lepton_flavor'], data['lepton_nChargedPf'], data['lepton_nPhotonPf'], data['lepton_nNeutralHadPf']])
  charged_pf_features = numpy.array([data['pf_charged_pt'], data['pf_charged_dR'], data['pf_charged_alpha'], data['pf_charged_ptRel'], data['pf_charged_puppiWeight'], data['pf_charged_fromPV'], data['pf_charged_pvAssociationQuality']])
  photon_pf_features = numpy.array([data['pf_photon_pt'], data['pf_photon_dR'], data['pf_photon_alpha'], data['pf_photon_ptRel'], data['pf_photon_puppiWeight']])
  f
  neutralHad_pf_features = numpy.array([data['pf_neutralHad_pt'], data['pf_neutralHad_dR'], data['pf_neutralHad_alpha'], data['pf_neutralHad_ptRel'], data['pf_neutralHad_puppiWeight']])
  #charged_pf_features = numpy.array([data['pf_charged_pt'], data['pf_charged_dR'], data['pf_charged_alpha'], data['pf_charged_puppiWeight'], data['pf_charged_fromPV'], data['pf_charged_pvAssociationQuality']])
  #photon_pf_features = numpy.array([data['pf_photon_pt'], data['pf_photon_dR'], data['pf_photon_alpha'], data['pf_photon_puppiWeight']])
  #neutralHad_pf_features = numpy.array([data['pf_neutralHad_pt'], data['pf_neutralHad_dR'], data['pf_neutralHad_alpha'], data['pf_neutralHad_puppiWeight']])

  label = data['lepton_isFromW']
  relIso = data['lepton_relIso03EA']

  # Preprocess
  for feature in global_features:
    feature = utils.preprocess(feature)

  for j in range(len(charged_pf_features)):
    charged_pf_features[j] = utils.preprocess_pf(charged_pf_features[j], j)

  for j in range(len(photon_pf_features)):
    photon_pf_features[j] = utils.preprocess_pf(photon_pf_features[j], j)

  for j in range(len(neutralHad_pf_features)):
    neutralHad_pf_features[j] = utils.preprocess_pf(neutralHad_pf_features[j], j)

  n_global_features = len(global_features)
  n_charged_pf_features = len(charged_pf_features)
  n_photon_pf_features = len(photon_pf_features)
  n_neutralHad_pf_features = len(neutralHad_pf_features)

  # Reorganize features
  global_features = numpy.transpose(global_features)
  charged_pf_features = utils.padArray(charged_pf_features, max_pf_charged)
  photon_pf_features = utils.padArray(photon_pf_features, max_pf_photon)
  neutralHad_pf_features = utils.padArray(neutralHad_pf_features, max_pf_neutralHad)

  type = 'train'
  if i < 5:
    type = 'test'

  f = h5py.File("prep/features_"+type+"_"+str(i)+".hdf5", "w")

  dset_global = f.create_dataset("global", data=global_features)
  dset_charged_pf = f.create_dataset("charged_pf", data=charged_pf_features)
  dset_photon_pf = f.create_dataset("photon_pf", data=photon_pf_features)
  dset_neutralHad_pf = f.create_dataset("neutralHad_pf", data=neutralHad_pf_features)
  dset_label = f.create_dataset("label", data=label)
  dset_relIso = f.create_dataset("relIso", data=relIso)

  f.close()
