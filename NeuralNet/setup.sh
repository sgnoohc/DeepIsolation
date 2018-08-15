cd /cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/cmssw/CMSSW_9_2_8/
cmsenv
cd -
#pip install --user -U scikit-learn
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages/:$PYTHONPATH

if [ ! -d weights ]; then
  mkdir weights
fi
if [ ! -d ROCs ]; then
  mkdir ROCs
fi
