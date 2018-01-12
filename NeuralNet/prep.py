import ROOT
import numpy
import root_numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

# Read data from ROOT file
f = ROOT.TFile('../BabyMaker/unknown_dummy.root')
tree = f.Get("t")

branches = ['lepton_isFromW', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_relIso03EA', 'lepton_chiso', 'lepton_nhiso', 'lepton_emiso', 'lepton_ncorriso', 'lepton_dxy', 'lepton_dz', 'lepton_ip3d', 'nvtx', 'lepton_flavor', 'pf_charged_pt', 'pf_charged_dR', 'pf_charged_ptRel', 'pf_charged_puppiWeight', 'pf_charged_fromPV', 'pf_charged_pvAssociationQuality', 'pf_photon_pt', 'pf_photon_dR', 'pf_photon_ptRel', 'pf_photon_puppiWeight', 'pf_neutralHad_pt', 'pf_neutralHad_dR', 'pf_neutralHad_ptRel', 'pf_neutralHad_puppiWeight']
data = root_numpy.tree2array(tree, branches = branches)

# Grab features
global_features = numpy.array([data['lepton_pt'], data['lepton_eta'], data['lepton_phi'], data['lepton_relIso03EA'], data['lepton_chiso'], data['lepton_nhiso'], data['lepton_emiso'], data['lepton_ncorriso'], data['lepton_dxy'], data['lepton_dz'], data['lepton_ip3d'], data['nvtx'], data['lepton_flavor']])
#charged_pf_features = numpy.array([data['pf_charged_pt'], data['pf_charged_dR'], data['pf_charged_ptRel'], data['pf_charged_puppiWeight'], data['pf_charged_fromPV'], data['pf_charged_pvAssociationQuality']])
#photon_pf_features = numpy.array([data['pf_photon_pt'], data['pf_photon_dR'], data['pf_photon_ptRel'], data['pf_photon_puppiWeight']])
f
#neutralHad_pf_features = numpy.array([data['pf_neutralHad_pt'], data['pf_neutralHad_dR'], data['pf_neutralHad_ptRel'], data['pf_neutralHad_puppiWeight']])
charged_pf_features = numpy.array([data['pf_charged_pt'], data['pf_charged_dR'], data['pf_charged_puppiWeight'], data['pf_charged_fromPV'], data['pf_charged_pvAssociationQuality']])
photon_pf_features = numpy.array([data['pf_photon_pt'], data['pf_photon_dR'], data['pf_photon_puppiWeight']])
f
neutralHad_pf_features = numpy.array([data['pf_neutralHad_pt'], data['pf_neutralHad_dR'], data['pf_neutralHad_puppiWeight']])
label = data['lepton_isFromW']
relIso = data['lepton_relIso03EA']

n_global_features = len(global_features)
n_charged_pf_features = len(charged_pf_features)
n_photon_pf_features = len(photon_pf_features)
n_neutralHad_pf_features = len(neutralHad_pf_features)

# Reorganize features
global_features = numpy.transpose(global_features)
charged_pf_features, charged_pf_timestep = utils.padArray(charged_pf_features)
photon_pf_features, photon_pf_timestep = utils.padArray(photon_pf_features)
neutralHad_pf_features, neutralHad_pf_timestep = utils.padArray(neutralHad_pf_features)

numpy.savez('features', global_features=global_features, charged_pf_features=charged_pf_features, photon_pf_features=photon_pf_features, neutralHad_pf_features=neutralHad_pf_features, label=label, relIso=relIso)
