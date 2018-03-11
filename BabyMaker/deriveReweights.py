import sys, os
import numpy

sys.path.insert(0,'BatchSubmit')
import corrupt 

from metis.Sample import DirectorySample

path = "/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/"

sample = DirectorySample(dataset = "ttbar", location = path)
corrupt_files = corrupt.find_corrupt_files(numpy.array([path[7:]]))
files = [f.name for f in sample.get_files() if f.name not in corrupt_files]

with open("deriveReweights.cpp", "w") as fout:
  fout.write('#include "ScanReweights.cpp"\n \n')
  fout.write('const TString savename = "weights_PtEta.root";\n')
  fout.write('int main(int argc, char* argv[]) {\n')
  fout.write('  TChain *ch = new TChain("Events");\n')
  for file in files:
    fout.write('  ch->Add("%s");\n' % file)
  fout.write('  ScanReweights(ch, -1, savename);\n')
  fout.write('}')

fout.close()
