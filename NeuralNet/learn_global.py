import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["KERAS_BACKEND"] = "tensorflow"

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
f = h5py.File('features_reweight.hdf5', 'r')

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

model = model.simple(n_global_features)

# Train & Test
nEpochs = 50*((2*10**6)//nTrain)
print('Training for %d epochs' % nEpochs)
nBatch = 10000

weights_file = "weights/"+savename+"_weights_{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weights_file) # save after every epoch 
callbacks_list = [checkpoint]


#model.load_weights("weights/Global_2mTrain_Reweight_weights_72.hdf5")
model.fit([global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch, callbacks=callbacks_list)

prediction = model.predict([global_features[nTrain:]], batch_size = nBatch)

prediction_training_set = model.predict([global_features[:nTrain]], batch_size = nBatch)

relIso = numpy.array(relIso)
relIso = relIso*(-1)

bdt_file = "../BDT/ROCs/BDT_GlobalOnly.npz"
npzfile_bdt = numpy.load(bdt_file)
fpr_bdt = npzfile_bdt['fpr']
tpr_bdt = npzfile_bdt['tpr']

fpr_re, tpr_re, thresh_re = metrics.roc_curve(label[nTrain:], relIso[nTrain:], pos_label = 1)
fpr_nn, tpr_nn, thresh_nn = metrics.roc_curve(label[nTrain:], prediction, pos_label = 1)

fpr_nn_train, tpr_nn_train, thresh_nn_train = metrics.roc_curve(label[:nTrain], prediction_training_set, pos_label=1)

numpy.savez('ROCs/RelIso', tpr=tpr_re, fpr=fpr_re)
numpy.savez('ROCs/'+savename, tpr_nn=tpr_nn, fpr_nn=fpr_nn)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.yaxis.set_ticks_position('both')
ax.grid(True)
plt.plot(fpr_re, tpr_re, color='darkred', lw=2, label='RelIso')
plt.plot(fpr_bdt, tpr_bdt, color='blue', lw=2, label='BDT')
plt.plot(fpr_nn, tpr_nn, color = 'darkorange', lw=2, label='DNN')
plt.xscale('log')
plt.grid()

plt.xlim([0.005, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (bkg. eff.)')
plt.ylabel('True Positive Rate (sig. eff.)')
plt.legend(loc='lower right')
plt.savefig('plot.pdf')

value1, idx1 = utils.find_nearest(tpr_nn, 0.90)
value2, idx2 = utils.find_nearest(tpr_nn, 0.99)
value3, idx3 = utils.find_nearest(fpr_nn, 0.01)
value4, idx4 = utils.find_nearest(fpr_nn, 0.1)

value1BDT, idx1BDT = utils.find_nearest(tpr_bdt, 0.90)
value2BDT, idx2BDT = utils.find_nearest(tpr_bdt, 0.99)
value3BDT, idx3BDT = utils.find_nearest(fpr_bdt, 0.01)
value4BDT, idx4BDT = utils.find_nearest(fpr_bdt, 0.1)

value1RE, idx1RE = utils.find_nearest(tpr_re, 0.90)
value2RE, idx2RE = utils.find_nearest(tpr_re, 0.99)
value3RE, idx3RE = utils.find_nearest(fpr_re, 0.01)
value4RE, idx4RE = utils.find_nearest(fpr_re, 0.1)


print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx1], tpr_nn[idx1]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx2], tpr_nn[idx2]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx3], tpr_nn[idx3]))
print('Neural net FPR, TPR: (%.3f, %.3f)' % (fpr_nn[idx4], tpr_nn[idx4]))

print('BDT FPR, TPR: (%.3f, %.3f)' % (fpr_bdt[idx1BDT], tpr_bdt[idx1BDT]))
print('BDT FPR, TPR: (%.3f, %.3f)' % (fpr_bdt[idx2BDT], tpr_bdt[idx2BDT]))
print('BDT FPR, TPR: (%.3f, %.3f)' % (fpr_bdt[idx3BDT], tpr_bdt[idx3BDT]))
print('BDT FPR, TPR: (%.3f, %.3f)' % (fpr_bdt[idx4BDT], tpr_bdt[idx4BDT]))

print('RE FPR, TPR: (%.3f, %.3f)' % (fpr_re[idx1RE], tpr_re[idx1RE]))
print('RE FPR, TPR: (%.3f, %.3f)' % (fpr_re[idx2RE], tpr_re[idx2RE]))
print('RE FPR, TPR: (%.3f, %.3f)' % (fpr_re[idx3RE], tpr_re[idx3RE]))
print('RE FPR, TPR: (%.3f, %.3f)' % (fpr_re[idx4RE], tpr_re[idx4RE]))

print('DeepIso AUC: %.5f' % metrics.auc(fpr_nn, tpr_nn))
print('DeepIso AUC (training set): %.5f' % metrics.auc(fpr_nn_train, tpr_nn_train))
print('RelIso AUC: %.5f' % metrics.auc(fpr_re, tpr_re))
                                                                                                        

