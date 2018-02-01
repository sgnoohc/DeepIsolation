### Search for and remove unhealthy hadoop files ###
### output of "hdfs fsck path-to-hadoop-dir" should be in a text file ###
import os

def find_corrupt_files(paths):
  corruptions = "corruptions.txt"

  os.system("touch %s" % corruptions)
  os.system("rm %s" % corruptions)
  for path in paths:
    os.system("hdfs fsck %s >> %s" % (path, corruptions))

  fnames = set([])
  with open(corruptions, "r") as fhin:
      for line in fhin:
	  if "CORRUPT" not in line: continue
	  if ".root" not in line: continue
	  fname = line.split()[0].replace(":","")
	  fnames.add("/hadoop"+fname)
  return fnames
