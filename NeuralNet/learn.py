import keras
import numpy
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npzfile = numpy.load('features.npz')

global_features = npzfile['global_features']
charged_pf_features = npzfile['charged_pf_features']
photon_pf_features = npzfile['photon_pf_features']
neutralHad_pf_features = npzfile['neutralHad_pf_features']
label = npzfile['label']
relIso = npzfile['relIso']

n_global_features = len(global_features[0])
n_charged_pf_features = len(charged_pf_features[0][0])
n_photon_pf_features = len(photon_pf_features[0][0])
n_neutralHad_pf_features = len(neutralHad_pf_features[0][0])

charged_pf_timestep = len(charged_pf_features[0])
photon_pf_timestep = len(photon_pf_features[0])
neutralHad_pf_timestep = len(neutralHad_pf_features[0])

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
nEpochs = 50
nBatch = 1000

model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch)
prediction = model.predict([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], batch_size = nBatch)

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
