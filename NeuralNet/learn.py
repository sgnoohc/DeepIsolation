import keras
import ROOT
import numpy
import root_numpy
from sklearn import metrics

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

n_global_features = len(global_features)
n_charged_pf_features = len(charged_pf_features)
n_photon_pf_features = len(photon_pf_features)
n_neutralHad_pf_features = len(neutralHad_pf_features)

# Reorganize features
global_features = numpy.transpose(global_features)
charged_pf_features, charged_pf_timestep = utils.padArray(charged_pf_features)
photon_pf_features, photon_pf_timestep = utils.padArray(photon_pf_features)
neutralHad_pf_features, neutralHad_pf_timestep = utils.padArray(neutralHad_pf_features)

# Structure NN
input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

lstm_charged_pf = keras.layers.LSTM(20)(input_charged_pf)
lstm_photon_pf = keras.layers.LSTM(10)(input_photon_pf)
lstm_neutralHad_pf = keras.layers.LSTM(10)(input_neutralHad_pf)

merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, input_global])
deep_layer = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform')(merged_features)
deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Train & Test
nTrain = 5000
nEpochs = 10
nBatch = 1000

model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch)
prediction = model.predict([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], batch_size = nBatch)

relIso = data['lepton_relIso03EA']
relIso = relIso[nTrain:]*(-1)

fpr_re, tpr_re, thresh = metrics.roc_curve(data['lepton_isFromW'][nTrain:], relIso, pos_label = 1)
fpr, tpr, thresh = metrics.roc_curve(data['lepton_isFromW'][nTrain:], prediction, pos_label = 1)
plt.figure()
plt.plot(fpr_re, tpr_re, color='darkred', lw=2, label='RelIso')
plt.plot(fpr, tpr, color = 'darkorange', lw=2, label='MLP')
plt.xscale('log')
plt.xlim([0.001, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('plot.pdf')
