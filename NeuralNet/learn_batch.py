import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
import numpy
import sys
from sklearn import metrics
import h5py
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import generator

if len(sys.argv) != 3:
  print('Incorrect number of arguments')
  print('Arg 1: save name')
  print('Arg 2: number of training events')
  exit(1)

savename = str(sys.argv[1])
nTrain = int(sys.argv[2])

f = h5py.File('prep/features_test_0.hdf5')

global_features = f['global']
charged_pf_features = f['charged_pf']
photon_pf_features = f['photon_pf']
neutralHad_pf_features = f['neutralHad_pf']
label = f['label']
relIso = f['relIso']

n_global_features = len(global_features[0])
n_charged_pf_features = len(charged_pf_features[0][0])
n_photon_pf_features = len(photon_pf_features[0][0])
n_neutralHad_pf_features = len(neutralHad_pf_features[0][0])

print(n_global_features)
print(n_charged_pf_features)
print(n_photon_pf_features)
print(n_neutralHad_pf_features)

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

lstm_charged_pf = keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1')(conv_charged_pf)
lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

lstm_photon_pf = keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1')(conv_photon_pf)
lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1')(conv_neutralHad_pf)
lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

# MLP to combine LSTM outputs with global features
dropout_rate = 0.25
merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, input_global])
deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_1')(merged_features)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_2')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_3')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_4')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_5')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_6')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_7')(deep_layer)
deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_7')(deep_layer)
output = keras.layers.Dense(1, activation = 'tanh', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Train & Test
nTrainAvailable = generator.nEvents_total(True)
nTestAvailable = generator.nEvents_total(False)
nTest = 500000
nEpochs = 1
nBatch = 10000

print("Training on %d of %d available training events" % (nTrain, nTrainAvailable))
print("Testing on %d of %d available testing events" % (nTest, nTestAvailable))
print("Training for %d epochs" % nEpochs)

model.fit_generator(generator = generator.generate(True, nTrain), steps_per_epoch = generator.nSteps(True, nTrain), epochs = nEpochs)
prediction = model.predict_generator(generator.generate(False, nTest), steps = generator.nSteps(False, nTest))

relIso = numpy.empty(shape=0) 
label = numpy.empty(shape=0)

files = glob.glob("prep/features_test_*.hdf5")
for file in files:
  f = h5py.File(file, "r")
  label = numpy.append(label, numpy.array(f['label']))
  relIso = numpy.append(relIso, numpy.array(f['relIso']))

label = label[:nTest]
relIso = relIso[:nTest]
relIso *= -1

npzfile_bdt = numpy.load('bdt_roc.npz')
fpr_bdt = npzfile_bdt['fpr']
tpr_bdt = npzfile_bdt['tpr']

fpr_re, tpr_re, thresh_re = metrics.roc_curve(label, relIso, pos_label = 1)
fpr_nn, tpr_nn, thresh_nn = metrics.roc_curve(label, prediction, pos_label = 1)

numpy.savez('ROCs/'+savename, tpr_nn=tpr_nn, fpr_nn=fpr_nn)

plt.figure()
plt.plot(fpr_re, tpr_re, color='darkred', lw=2, label='RelIso')
plt.plot(fpr_bdt, tpr_bdt, color='aqua', lw=2, label='BDT trained w/sum. vars')
plt.plot(fpr_nn, tpr_nn, color = 'darkorange', lw=2, label='DeepIsolation')
plt.xscale('log')
plt.xlim([0.001, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('plot.pdf')

value1, idx1 = utils.find_nearest(fpr_nn, 0.087)
value2, idx2 = utils.find_nearest(tpr_nn, 0.817)

print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx1], tpr_nn[idx1]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx2], tpr_nn[idx2]))

