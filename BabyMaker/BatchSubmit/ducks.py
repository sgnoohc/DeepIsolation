import sys, os
import time
import itertools
import numpy

from metis.Sample import DirectorySample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser

job_tag = "DeepIso_v0.0.5"
exec_path = "condor_exe.sh"
tar_path = "package.tar.gz"
hadoop_path = "DeepIsolation"

import corrupt

os.system("rm processBaby")
os.system("rm package.tar.gz")
os.system("cp ../processBaby .") # for some reason this doesn't like to update
os.system("cp ../weights_PtEta.root .")
os.system("tar -czf package.tar.gz processBaby weights_PtEta.root CORE")

dslocs = [
    [ "/TTbar", "/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] , 
#    [ "/TTJets_SingleLeptFromT", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] ,
#    [ "/TTJets_SingleLeptFromT_ext1", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_SingleLeptFromTBar", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] ,
#    [ "/TTJets_SingleLeptFromTBar_ext1", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_DiLept", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] ,
#    [ "/TTJets_DiLept_ext1", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_HT-600to800", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_HT-800to1200", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_HT-800to1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_HT-1200to2500", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTJets_HT-2500toInf", "/hadoop/cms/store/group/snt/run2_moriond17/TTJets_HT-2500toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/V08-00-16/" ] ,
#    [ "/TTTo2L2Nu", "/hadoop/cms/store/group/snt/run2_moriond17/TTTo2L2Nu_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] ,
]

total_summary = {}
while True:
    allcomplete = True
    for ds,loc in dslocs:
        sample = DirectorySample( dataset=ds, location=loc )
        corrupt_files = corrupt.find_corrupt_files(numpy.array([loc[7:]]))
        files = [f.name for f in sample.get_files() if f.name not in corrupt_files]
        sample.set_files(files)
        task = CondorTask(
                sample = sample,
                open_dataset = False,
                flush = True,
                files_per_output = 10,
                output_name = "merged_ntuple.root",
                tag = job_tag,
                cmssw_version = "CMSSW_9_2_1", # doesn't do anything
                executable = exec_path,
                tarfile = tar_path,
                condor_submit_params = {"sites" : "T2_US_UCSD"},
                special_dir = hadoop_path
                )
        task.process()
        allcomplete = allcomplete and task.complete()
        # save some information for the dashboard
        total_summary[ds] = task.get_task_summary()
    # parse the total summary and write out the dashboard
    StatsParser(data=total_summary, webdir="~/public_html/dump/deepiso/").do()
    os.system("chmod -R 755 ~/public_html/dump/deepiso")
    if allcomplete:
        print ""
        print "Job={} finished".format(job_tag)
        print ""
        break
    print "Sleeping 300 seconds ..."
    time.sleep(300)
