PACKAGE=package.tar.gz
OUTPUTDIR=$1
OUTPUTFILENAME=$2
INPUTFILENAMES=$3
INDEX=$4

echo "OUTPUTDIR : $OUTPUTDIR"
echo "OUTPUTFILENAME : $OUTPUTFILENAME"
echo "INPUTFILENAMES : $INPUTFILENAMES"
echo "INDEX : $INDEX"

echo "[wrapper] printing env"
printenv
echo 

echo "[wrapper] hostname  = " `hostname`
echo "[wrapper] date      = " `date`
echo "[wrapper] linux timestamp = " `date +%s`

export SCRAM_ARCH=slc6_amd64_gcc530

source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/CMSSW_9_2_0/src
eval `scramv1 runtime -sh`

popd

tar -xvf package.tar.gz

./processBaby "${INPUTFILENAMES}" "${OUTPUTFILENAME}" "${INDEX}"

ls -l

export LD_PRELOAD=/usr/lib64/gfal2-plugins//libgfal_plugin_xrootd.so # needed in CMSSW_10 (and some late CMSSW_9)
gfal-copy -p -f -t 4200 --verbose file://`pwd`/${OUTPUTFILENAME}_${INDEX}.root gsiftp://gftp.t2.ucsd.edu/${OUTPUTDIR}/${OUTPUTFILENAME}_${INDEX}.root --checksum ADLER32

