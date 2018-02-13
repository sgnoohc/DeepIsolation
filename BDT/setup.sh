source /code/osgcode/cmssoft/cmsset_default.sh  > /dev/null 2>&1
export SCRAM_ARCH=slc6_amd64_gcc530
export CMSSW_VERSION=CMSSW_8_0_3
echo Setting up CMSSW for CMSSW_8_0_3 for slc6_amd64_gcc530
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/cmssw/CMSSW_8_0_3/src
eval `scramv1 runtime -sh`
cd -
