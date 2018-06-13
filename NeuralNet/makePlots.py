import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

from sklearn.metrics import auc

#bdt_file = "../BDT/ROCs/BDT_GlobalNoKin.npz"
#bdt_file = "../BDT/ROCs/BDT_GlobalNoIP.npz" # or change to desired BDT file with fpr and tpr numpy arrays
#bdt_file = "../BDT/ROCs/BDT_Global_NoIPRoundedKinematics.npz"
bdt_file = "../BDT/ROCs/BDT_AllFeatures_RoundedKinematics.npz"
npzfile_bdt = numpy.load(bdt_file)
fpr_bdt = npzfile_bdt['fpr']
tpr_bdt = npzfile_bdt['tpr']

re_file = "ROCs/RelIso.npz"
npzfile_re = numpy.load(re_file)
fpr_re = npzfile_re['fpr']
tpr_re = npzfile_re['tpr']

nn_file = "ROCs/GlobalNoIP_2mTrain.npz"
#npzfile_nn = numpy.load(nn_file)
#fpr_nn = npzfile_nn['fpr_nn']
#tpr_nn = npzfile_nn['tpr_nn']

nn_1_file = "ROCs/GlobalOnly_1p5mTrain.npz"
#npzfile_nn_1 = numpy.load(nn_1_file)
#fpr_nn_1 = npzfile_nn_1['fpr_nn']
#tpr_nn_1 = npzfile_nn_1['tpr_nn']

nn_2_file = "ROCs/CategorizedGlobalOnly_1p5mTrain_4Output.npz"
#npzfile_nn_2 = numpy.load(nn_2_file)
#fpr_nn_2 = npzfile_nn_2['fpr_nn']
#tpr_nn_2 = npzfile_nn_2['tpr_nn']

nn_3_file = "ROCs/CategorizedGlobalOnlyNoKin_1p5mTrain.npz"
#npzfile_nn_3 = numpy.load(nn_3_file)
#fpr_nn_3 = npzfile_nn_3['fpr_nn']
#tpr_nn_3 = npzfile_nn_3['tpr_nn']

nn_4_file = "ROCs/CategorizedGlobalOnly_1p5mTrain_1IsoOutput.npz"
#npzfile_nn_4 = numpy.load(nn_4_file)
#fpr_nn_4 = npzfile_nn_4['fpr_nn']
#tpr_nn_4 = npzfile_nn_4['tpr_nn']

nn_5_file = "ROCs/CategorizedGlobalOnly_1p5mTrain_4IsoOutput.npz"
#npzfile_nn_5 = numpy.load(nn_5_file)
#fpr_nn_5 = npzfile_nn_5['fpr_nn']
#tpr_nn_5 = npzfile_nn_5['tpr_nn']

nn_6_file = "ROCs/GlobalOnly_NoIP_RoundedKinematics_1p5mTrain.npz"
#npzfile_nn_6 = numpy.load(nn_6_file)
#fpr_nn_6 = npzfile_nn_6['fpr_nn']
#tpr_nn_6 = npzfile_nn_6['tpr_nn']

nn_7_file = "ROCs/AllFeatures_RoundedKinematics_2mTrain.npz"
npzfile_nn_7 = numpy.load(nn_7_file)
fpr_nn_7 = npzfile_nn_7['fpr_nn']
tpr_nn_7 = npzfile_nn_7['tpr_nn']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.yaxis.set_ticks_position('both')
ax.grid(True)
plt.grid(color='black', linestyle='--', linewidth = 0.1, which = 'both')
plt.plot(fpr_re, tpr_re, color='darkred', lw=3, label='RelIso')
plt.plot(fpr_bdt, tpr_bdt, color='blue', lw=3, label='BDT')
#plt.plot(fpr_nn_3, tpr_nn_3, color = 'darkorange', lw=3, label='DNN')
#plt.plot(fpr_nn_1, tpr_nn_1, color = 'darkorange', lw=3, label='DNN (Standard)')
#plt.plot(fpr_nn_2, tpr_nn_2, color = 'green', lw=3, label='DNN (Categorized)')
#plt.plot(fpr_nn_4, tpr_nn_4, color = 'darkorange', lw=3, label='DNN (1 isolation output)')
#plt.plot(fpr_nn_5, tpr_nn_5, color = 'green', lw=3, label='DNN (4 isolation output)')
#plt.plot(fpr_nn_6, tpr_nn_6, color = 'darkorange', lw=3, label='DNN')
plt.plot(fpr_nn_7, tpr_nn_7, color = 'darkorange', lw=3, label='DNN')
plt.xscale('log')

plt.xlim([0.005, 1.0])
plt.ylim([0.3, 1.05])
plt.xlabel('False Positive Rate (background efficiency)')
plt.ylabel('True Positive Rate (signal efficiency)')
plt.legend(loc='lower right')
plt.savefig('plot.pdf', bbox_inches='tight')

tpr_nn = tpr_nn_7
fpr_nn = fpr_nn_7

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
