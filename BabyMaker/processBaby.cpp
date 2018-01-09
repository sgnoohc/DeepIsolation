#include <iostream>

#include "ScanChain.cpp"

int main(int argc, char* argv[])
{
  TString outfileid(argv[1]); 
  TString infile(argv[2]); 

  int max_events = -1;
  if (argc >= 4) max_events = atoi(argv[3]);
  std::cout << "set max number of events to: " << max_events << std::endl;
  
  std::cout<<"running on file: "<<infile.Data()<<std::endl;
  
  TChain *chain = new TChain("Events");
  //chain->Add(infile.Data());
  chain->Add("/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/merged_ntuple_1.root");
  if (chain->GetEntries() == 0) {
    std::cout << "ERROR: no entries in chain. filename was: " << infile << std::endl;
    return 2;
  }


  char* sample;
  //MC
  if (infile.Contains("TTJets_MassiveBinDECAY_TuneZ2star_8TeV"))     sample = Form("ttall_%s",  	 outfileid.Data());
  else sample = Form("unknown_%s", outfileid.Data());

  std::cout<<"sample is "<<sample<<std::endl;
  std::cout<<"outfileid "<<outfileid.Data()<<std::endl;

  //--------------------------------
  // run
  //--------------------------------
  
  BabyMaker *looper = new BabyMaker();
  looper->ScanChain(chain, sample, max_events); 
  return 0;
}
