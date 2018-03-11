source /code/osgcode/cmssoft/cmsset_default.sh > /dev/null 2>&1
export SCRAM_ARCH=slc6_amd64_gcc530   # or whatever scram_arch you need for your desired CMSSW release
export CMSSW_VERSION=CMSSW_9_2_0
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd -

if [ ! -d "CORE" ]; then
  ln -s ../CORE/ .
fi
if [ ! -d "BatchSubmit/CORE" ]; then
  cp -R ../CORE/ BatchSubmit/
fi

if [ -d "~/ProjectMetis/" ]; then
  pushd ~/ProjectMetis/
  source setup.sh
  popd
fi
