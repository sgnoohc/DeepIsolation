import root_numpy
import ROOT
import numpy
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = ["BDT_GlobalOnly.root", "BDT_CandsOnly.root", "BDT_AllFeatures.root", "BDT_GlobalNoIP.root", "BDT_GlobalNoKin.root", "BDT_Global_NoIPRoundedKinematics.root", "BDT_AllFeatures_RoundedKinematics.root"]

branches = ['classID', 'BDT']

for file in files:
  f = ROOT.TFile(file)
  tree = f.Get("TestTree")

  data = root_numpy.tree2array(tree, branches = branches)

  tpr, fpr, thresh = metrics.roc_curve(data['classID'], data['BDT']) # labels are interpreted backwards (that's why tpr and fpr are reversed)
  numpy.savez('ROCs/' + file.replace(".root",""), tpr=tpr, fpr=fpr)

  plt.figure()
  plt.plot(tpr, fpr, color = 'aqua', label ='BDT')
  plt.xscale('log')
  plt.xlim([0.001, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.savefig('plot' + file.replace(".root","") + '.pdf')
