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
files = glob.glob('/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.10/merged_ntuple_*.root') # rounded kinematics
#files = ['../BabyMaker/dummyJob_123.root']


branches = ['lepton_isFromW', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_relIso03EA', 'lepton_chiso', 'lepton_nhiso', 'lepton_emiso', 'lepton_ncorriso', 'lepton_dxy', 'lepton_dz', 'lepton_ip3d', 'nvtx', 'lepton_flavor', 'lepton_nChargedPf', 'lepton_nPhotonPf', 'lepton_nNeutralHadPf', 'pf_image', 'pf_image_charged', 'pf_image_photon', 'pf_image_neutralHadron']
#branches = ['lepton_isFromW', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_relIso03EA', 'lepton_chiso', 'lepton_nhiso', 'lepton_emiso', 'lepton_ncorriso', 'lepton_dxy', 'lepton_dz', 'lepton_ip3d', 'nvtx', 'lepton_flavor', 'lepton_nChargedPf', 'lepton_nPhotonPf', 'lepton_nNeutralHadPf', 'lepton_nOuterPf', 'pf_charged_pt', 'pf_charged_dR', 'pf_charged_alpha', 'pf_charged_pPRel', 'pf_charged_puppiWeight', 'pf_charged_fromPV', 'pf_charged_pvAssociationQuality', 'pf_photon_pt', 'pf_photon_dR', 'pf_photon_alpha', 'pf_photon_pPRel', 'pf_photon_puppiWeight', 'pf_neutralHad_pt', 'pf_neutralHad_dR', 'pf_neutralHad_alpha', 'pf_neutralHad_pPRel', 'pf_neutralHad_puppiWeight', 'pf_outer_pt', 'pf_outer_dR', 'pf_outer_alpha', 'pf_outer_pPRel', 'pf_outer_type', 'substr_ptrel', 'substr_sumdij', 'substr_nreclsj', ('substr_reclsj_dr', 0), ('substr_reclsj_pt', 0), 'pf_annuli_energy[0]', 'pf_annuli_energy[1]', 'pf_annuli_energy[2]', 'pf_annuli_energy[3]','pf_annuli_energy[4]','pf_annuli_energy[5]','pf_annuli_energy[6]','pf_annuli_energy[7]']

nFiles = 1 # change this as needed
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
global_features = numpy.array([data['lepton_pt'], data['lepton_eta'], data['lepton_phi'], data['lepton_relIso03EA'], data['lepton_chiso'], data['lepton_nhiso'], data['lepton_emiso'], data['lepton_ncorriso'], data['nvtx'], data['lepton_flavor'], data['lepton_nChargedPf'], data['lepton_nPhotonPf'], data['lepton_nNeutralHadPf']]) # no IP
image_features = data['pf_image']
image_charged_features = data['pf_image_charged']
image_photon_features = data['pf_image_photon']
image_neutralHadron_features = data['pf_image_neutralHadron']

label = data['lepton_isFromW']
relIso = data['lepton_relIso03EA']

# Preprocess
for feature in global_features:
  feature = utils.preprocess(feature)

image_features = utils.reform_image(image_features)
image_features_color = utils.reform_image_color(image_charged_features, image_photon_features, image_neutralHadron_features)

n_global_features = len(global_features)

# Reorganize features
global_features = numpy.transpose(global_features)

f = h5py.File("features_image.hdf5", "w") # change name as needed

dset_global = f.create_dataset("global", data=global_features)
dset_image = f.create_dataset("image", data=image_features)
dset_image_color = f.create_dataset("image_color", data=image_features_color)
dset_label = f.create_dataset("label", data=label)
dset_relIso = f.create_dataset("relIso", data=relIso)

f.close()
