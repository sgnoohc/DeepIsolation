source /code/osgcode/cmssoft/cmsset_default.sh > /dev/null 2>&1
export SCRAM_ARCH=slc6_amd64_gcc530   # or whatever scram_arch you need for your desired CMSSW release
export CMSSW_VERSION=CMSSW_9_2_0
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd -

if [ ! -d "CORE" ]; then
  cp -R ../CORE/ .
fi
if [ ! -d "BatchSubmit/CORE" ]; then
  cp -R ../CORE/ BatchSubmit/
fi

if [ -d ~/ProjectMetis/ ]; then
  pushd ~/ProjectMetis/
  echo "Setting up Metis"
  source setup.sh
  popd
fi

# Make CMS3 class files
# Note: if you want to use different CMS3/4 ntuples, you'll need to manually change the path to the ntuple 
git clone https://github.com/cmstas/software.git
cd software/makeCMS3ClassFiles/
root -l -b -q 'makeCMS3ClassFiles.C+("/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/merged_ntuple_1.root", "Events")'
cp CMS3* ../../
