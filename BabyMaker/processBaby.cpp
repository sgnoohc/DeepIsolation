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
  chain->Add(infile.Data());
  // chain->Add("/hadoop/cms/store/group/snt/run2_50ns/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/V07-04-03/merged_ntuple_2.root");
  // chain->Add("/hadoop/cms/store/group/snt/run2_50ns/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/V07-04-03/merged_ntuple_3.root");
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
