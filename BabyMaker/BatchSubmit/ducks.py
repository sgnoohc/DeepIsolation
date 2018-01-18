import sys, os

from metis.Sample import DirectorySample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser

job_tag = "DeepIsolation_Babies"
exec_path = "condor_exe.sh"
tar_path = "package.tar.gz"
hadoop_path = "DeepIsolation"

os.system("tar -czf package.tar.gz ../processBaby")

dslocs = [
    [ "/TTbar", "/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/" ] , 
]

total_summary = {}
while True:
    allcomplete = True
    for ds,loc in dslocs:
        task = CondorTask(
                sample = DirectorySample( dataset=ds, location=loc ),
                open_dataset = False,
                flush = True,
                files_per_output = 5,
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
