# BabyMaker 
Makes baby ntuples from CMS3 to feed into NN.

### Running
1. `cd BabyMaker ; source setup.sh`
2. Create looper file for deriving pT and eta reweighting: `python deriveReweights.py`
3. `make`
4. Create file containing pT/eta acceptance probabilities for reweighting: `./deriveReweights`
5. Run a test case baby: `./processBaby /hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/merged_ntuple_1.root dummyJob 123 [nevents=-1]`
6. When satisfied with your test case, submit full set of jobs to condor. `cd BatchSubmit ; python ducks.py`

### Metis
Batch submission relies on Metis, and assumes that you have Metis set up in your home directory. If you need to set up Metis, you can get it from here: `https://github.com/aminnj/ProjectMetis`

### Modifying babies
To add a variable to the babies, add it to the `BabyMaker` class in `ScanChain.h` and implement it in `ScanChain.cpp`.
