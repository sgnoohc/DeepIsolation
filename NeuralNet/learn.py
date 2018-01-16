import keras
import numpy
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

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

################
# Structure NN #
################

# Inputs
input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

# Convolutional layers for pf cands
dropout_rate = 0.1
conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_1')(conv_charged_pf)
conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_2')(conv_charged_pf)
conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_3')(conv_charged_pf)
conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_1')(conv_photon_pf)
conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_2')(conv_photon_pf)
conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_3')(conv_photon_pf)

conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_1')(conv_neutralHad_pf)
conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_2')(conv_neutralHad_pf)
conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_3')(conv_neutralHad_pf)

# LSTMs for pf cands
batch_momentum = 0.6

lstm_charged_pf = keras.layers.LSTM(100, implementation = 2, name ='lstm_charged_pf_1')(conv_charged_pf)
lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

lstm_photon_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_photon_pf_1')(conv_photon_pf)
lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1')(conv_neutralHad_pf)
lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

# MLP to combine LSTM outputs with global features
merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, input_global])
deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform')(merged_features)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform')(deep_layer)
output = keras.layers.Dense(1, activation = 'tanh', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Train & Test
nTrain = 400000
nEpochs = 15
nBatch = 10000


model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch)
prediction = model.predict([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], batch_size = nBatch)

relIso = relIso[nTrain:]*(-1)

fpr_re, tpr_re, thresh_re = metrics.roc_curve(label[nTrain:], relIso, pos_label = 1)
fpr_nn, tpr_nn, thresh_nn = metrics.roc_curve(label[nTrain:], prediction, pos_label = 1)
plt.figure()
plt.plot(fpr_re, tpr_re, color='darkred', lw=2, label='RelIso')
plt.plot(fpr_nn, tpr_nn, color = 'darkorange', lw=2, label='DeepIsolation')
plt.xscale('log')
plt.xlim([0.001, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('plot.pdf')

#value, idx = utils.find_nearest(thresh_re, 0.06)
#print('RelIso = 0.06 cut: (%.3f, %.3f)' % (fpr_re[idx], tpr_re[idx]))
#value_nn, idx_nn = utils.find_nearest(tpr_nn, tpr_re[idx])
#print('Neural net same TPR: (%.3f, %.3f)' % (fpr_nn[idx_nn], tpr_nn[idx_nn]))

value1, idx1 = utils.find_nearest(fpr_nn, 0.087)
value2, idx2 = utils.find_nearest(tpr_nn, 0.817)

print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx1], tpr_nn[idx1]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx2], tpr_nn[idx2]))

