import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
import numpy
import sys
from sklearn import metrics
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import model

# Parse args
if len(sys.argv) != 3:
  print('Incorrect number of arguments')
  print('Arg 1: save name')
  print('Arg 2: number of training events')
  exit(1)

savename = str(sys.argv[1])
nTrain = int(sys.argv[2])

# Read features from hdf5 file
f = h5py.File('features_els_11m.hdf5', 'r')

global_features = f['global']
charged_pf_features = f['charged_pf']
photon_pf_features = f['photon_pf']
neutralHad_pf_features = f['neutralHad_pf']
label = f['label']
relIso = f['relIso']

#global_features = numpy.transpose(numpy.array([relIso])) # uncomment this line to train with only pf cands + RelIso

n_global_features = len(global_features[0])
n_charged_pf_features = len(charged_pf_features[0][0])
n_photon_pf_features = len(photon_pf_features[0][0])
n_neutralHad_pf_features = len(neutralHad_pf_features[0][0])

charged_pf_timestep = len(charged_pf_features[0])
photon_pf_timestep = len(photon_pf_features[0])
neutralHad_pf_timestep = len(neutralHad_pf_features[0])

print(n_global_features)
print(n_charged_pf_features)
print(n_photon_pf_features)
print(n_neutralHad_pf_features)
print(len(label))

print(charged_pf_timestep)
print(photon_pf_timestep)
print(neutralHad_pf_timestep)

################
# Structure NN #
################

# Inputs
input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

model = model.base(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features)

# Train & Test
nEpochs = 100
nBatch = 10000

weights_file = "weights/"+savename+"_weights_{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weights_file) # save after every epoch 
callbacks_list = [checkpoint]

#validation_data = ([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], label[nTrain:])


print(model.summary())
#model.load_weights(weights_file)
#model.load_weights("weights/Global_10MTrain_Els_weights_09.hdf5")
model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch, callbacks=callbacks_list)
#model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch)
prediction = model.predict([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], batch_size = nBatch)

prediction_training_set = model.predict([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], batch_size = nBatch) 

relIso = numpy.array(relIso)
relIso = relIso*(-1)

npzfile_bdt = numpy.load('bdt_roc.npz')
fpr_bdt = npzfile_bdt['fpr']
tpr_bdt = npzfile_bdt['tpr']

fpr_re, tpr_re, thresh_re = metrics.roc_curve(label[nTrain:], relIso[nTrain:], pos_label = 1)
fpr_nn, tpr_nn, thresh_nn = metrics.roc_curve(label[nTrain:], prediction, pos_label = 1)

fpr_nn_train, tpr_nn_train, thresh_nn_train = metrics.roc_curve(label[:nTrain], prediction_training_set, pos_label=1)

numpy.savez('ROCs/RelIso', tpr=tpr_re, fpr=fpr_re)
numpy.savez('ROCs/'+savename, tpr_nn=tpr_nn, fpr_nn=fpr_nn)

plt.figure()
plt.plot(fpr_re, tpr_re, color='darkred', lw=2, label='RelIso')
#plt.plot(fpr_bdt, tpr_bdt, color='aqua', lw=2, label='BDT trained w/sum. vars')
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

value3, idx3 = utils.find_nearest(fpr_nn, 0.01)
value4, idx4 = utils.find_nearest(fpr_nn, 0.1)

print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx1], tpr_nn[idx1]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx2], tpr_nn[idx2]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx3], tpr_nn[idx3]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx4], tpr_nn[idx4]))
print('DeepIso AUC: %.5f' % metrics.auc(fpr_nn, tpr_nn))
print('DeepIso AUC (training set): %.5f' % metrics.auc(fpr_nn_train, tpr_nn_train))
print('RelIso AUC: %.5f' % metrics.auc(fpr_re, tpr_re))
