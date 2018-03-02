import ROOT
import numpy
import root_numpy
import h5py
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

# Read data from ROOT file
files = glob.glob('/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.3/merged_ntuple_*.root')

branches = ['lepton_isFromW', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_relIso03EA', 'lepton_chiso', 'lepton_nhiso', 'lepton_emiso', 'lepton_ncorriso', 'lepton_dxy', 'lepton_dz', 'lepton_ip3d', 'nvtx', 'lepton_flavor', 'lepton_nChargedPf', 'lepton_nPhotonPf', 'lepton_nNeutralHadPf', 'pf_charged_pt', 'pf_charged_dR', 'pf_charged_alpha', 'pf_charged_pPRel', 'pf_charged_puppiWeight', 'pf_charged_fromPV', 'pf_charged_pvAssociationQuality', 'pf_photon_pt', 'pf_photon_dR', 'pf_photon_alpha', 'pf_photon_pPRel', 'pf_photon_puppiWeight', 'pf_neutralHad_pt', 'pf_neutralHad_dR', 'pf_neutralHad_alpha', 'pf_neutralHad_pPRel', 'pf_neutralHad_puppiWeight', 'pf_outer_pt', 'pf_outer_dR', 'pf_outer_alpha', 'pf_outer_pPRel', 'pf_outer_type', 'substr_ptrel', 'substr_sumdij', 'substr_nreclsj', ('substr_reclsj_dr', 0), ('substr_reclsj_pt', 0)]

nFiles = 32 # change this as needed. Roughly 300k muons per file
data = numpy.empty(shape=0)

idx = 0
for file in files:
  if idx >= nFiles:
    break

  f = ROOT.TFile(file)
  tree = f.Get("t")
  if len(data) == 0:
    #data = root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 0') # electrons
    data = root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 1') # muons
  else:
    #data = numpy.append(data, root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 0'))
    data = numpy.append(data, root_numpy.tree2array(tree, branches = branches, selection = 'lepton_flavor == 1'))

  print(len(data['lepton_isFromW']))
  idx += 1

# Grab features
global_features = numpy.array([data['lepton_pt'], data['lepton_eta'], data['lepton_phi'], data['lepton_relIso03EA'], data['lepton_chiso'], data['lepton_nhiso'], data['lepton_emiso'], data['lepton_ncorriso'], data['lepton_dxy'], data['lepton_dz'], data['lepton_ip3d'], data['nvtx'], data['lepton_flavor'], data['lepton_nChargedPf'], data['lepton_nPhotonPf'], data['lepton_nNeutralHadPf']])
#global_features = numpy.array([data['lepton_pt'], data['lepton_eta'], data['lepton_phi'], data['lepton_relIso03EA'], data['lepton_chiso'], data['lepton_nhiso'], data['lepton_emiso'], data['lepton_ncorriso'], data['lepton_dxy'], data['lepton_dz'], data['lepton_ip3d'], data['nvtx'], data['lepton_flavor'], data['lepton_nChargedPf'], data['lepton_nPhotonPf'], data['lepton_nNeutralHadPf'], data['substr_ptrel'], data['substr_sumdij'], data['substr_nreclsj'], data['substr_reclsj_dr'], data['substr_reclsj_pt']])
charged_pf_features = numpy.array([data['pf_charged_pt'], data['pf_charged_dR'], data['pf_charged_alpha'], data['pf_charged_pPRel'], data['pf_charged_puppiWeight'], data['pf_charged_fromPV'], data['pf_charged_pvAssociationQuality']])
photon_pf_features = numpy.array([data['pf_photon_pt'], data['pf_photon_dR'], data['pf_photon_alpha'], data['pf_photon_pPRel'], data['pf_photon_puppiWeight']])
neutralHad_pf_features = numpy.array([data['pf_neutralHad_pt'], data['pf_neutralHad_dR'], data['pf_neutralHad_alpha'], data['pf_neutralHad_pPRel'], data['pf_neutralHad_puppiWeight']])
outer_pf_features = numpy.array([data['pf_outer_pt'], data['pf_outer_dR'], data['pf_outer_alpha'], data['pf_outer_pPRel'], data['pf_outer_type']])

label = data['lepton_isFromW']
relIso = data['lepton_relIso03EA']

# Preprocess
for feature in global_features:
  feature = utils.preprocess(feature)

nChargedCutoff = 15
nPhotonCutoff = 5
nNeutralHadCutoff = 3
nOuterCutoff = 30

for i in range(len(charged_pf_features)):
  charged_pf_features[i] = utils.preprocess_pf(charged_pf_features[i], nChargedCutoff)

for i in range(len(photon_pf_features)):
  photon_pf_features[i] = utils.preprocess_pf(photon_pf_features[i], nPhotonCutoff)

for i in range(len(neutralHad_pf_features)):
  neutralHad_pf_features[i] = utils.preprocess_pf(neutralHad_pf_features[i], nNeutralHadCutoff)

for i in range(len(outer_pf_features)):
  outer_pf_features[i] = utils.preprocess_pf(outer_pf_features[i], nOuterCutoff)

n_global_features = len(global_features)
n_charged_pf_features = len(charged_pf_features)
n_photon_pf_features = len(photon_pf_features)
n_neutralHad_pf_features = len(neutralHad_pf_features)
n_outer_pf_features = len(outer_pf_features)

# Reorganize features
global_features = numpy.transpose(global_features)
charged_pf_features = utils.padArray(charged_pf_features, nChargedCutoff)
photon_pf_features = utils.padArray(photon_pf_features, nPhotonCutoff)
neutralHad_pf_features = utils.padArray(neutralHad_pf_features, nNeutralHadCutoff)
outer_pf_features = utils.padArray(outer_pf_features, nOuterCutoff)

f = h5py.File("features_els_11m.hdf5", "w")

dset_global = f.create_dataset("global", data=global_features)
dset_charged_pf = f.create_dataset("charged_pf", data=charged_pf_features)
dset_photon_pf = f.create_dataset("photon_pf", data=photon_pf_features)
dset_neutralHad_pf = f.create_dataset("neutralHad_pf", data=neutralHad_pf_features)
dset_outer_pf = f.create_dataset("outer_pf", data=outer_pf_features)
dset_label = f.create_dataset("label", data=label)
dset_relIso = f.create_dataset("relIso", data=relIso)

f.close()
