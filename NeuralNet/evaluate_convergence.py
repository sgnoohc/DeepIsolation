### evaluate_convergence.py
### This program trains a DeepIsolation model (delivered by model.py) and examines the convergence by plotting training/testing AUC as a function of number of epcohs


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
f = h5py.File('features.hdf5', 'r')

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

model = model.base(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features)

# Train & Test
nEpochs = 1
nBatch = 2500

weights_file = "weights/"+savename+"_weights_{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weights_file) # save after every epoch 
callbacks_list = [checkpoint]

#validation_data = ([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], label[nTrain:])

#model.load_weights(weights_file)
model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch, callbacks=callbacks_list)


relIso = numpy.array(relIso)
relIso = relIso*(-1)

auc_test = numpy.zeros(nEpochs)
auc_train = numpy.zeros(nEpochs)
x = numpy.linspace(1, nEpochs, nEpochs)

for i in range(nEpochs):
  print(i)
  model.load_weights("weights/"+savename+"_weights_" + str(i+1).zfill(2) + ".hdf5")
  prediction = model.predict([charged_pf_features[nTrain:], photon_pf_features[nTrain:], neutralHad_pf_features[nTrain:], global_features[nTrain:]], batch_size = nBatch)
  prediction_training_set = model.predict([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], batch_size = nBatch) 
 

  plt.figure()
  plt.hist(prediction[label[nTrain:].astype(bool)], bins = 100, label = 'Signal', color = 'red')
  plt.hist(prediction[numpy.logical_not(label[nTrain:])], bins = 100, label = 'Background', color = 'blue')
  plt.legend(loc = 'upper left')
  plt.xlabel('DeepIsolation Discriminant')
  plt.ylabel('Events')
  plt.savefig("weights/discriminant_" + str(i+1).zfill(2) + ".pdf") 
  plt.clf()

  fpr_nn, tpr_nn, thresh_nn = metrics.roc_curve(label[nTrain:], prediction, pos_label = 1)
  fpr_nn_train, tpr_nn_train, thresh_nn_train = metrics.roc_curve(label[:nTrain], prediction_training_set, pos_label=1)

  auc_test[i] = metrics.auc(fpr_nn, tpr_nn)
  auc_train[i] = metrics.auc(fpr_nn_train, tpr_nn_train)

plt.figure()
plt.plot(x, auc_test, color = 'red', label = 'Testing')
plt.plot(x, auc_train, color = 'blue', label = 'Training')
#plt.plot(x, numpy.ones_like(x)*0.977, 'b-', label = 'BDT')
#plt.plot(x, numpy.ones_like(x)*0.922, 'r-', label = 'RelIso')
plt.ylim([0.9,1.0])
plt.legend(loc = 'upper left')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.savefig("convergence.pdf")

numpy.savez('weights/convergence_' + savename, auc_test = auc_test, auc_train = auc_train)

print('Best testing AUC: %.5f' % max(auc_test))
print('Corresponding training AUC: %.5f' % auc_train[auc_test.argmax()])
