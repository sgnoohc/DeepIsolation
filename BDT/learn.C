#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"

void learn(int nTrain)
{
  // Initialize TMVA
  TMVA::Tools::Instance();

  TFile* outputFile = TFile::Open("BDT.root", "RECREATE");

  TMVA::Factory *factory = new TMVA::Factory("TMVA", outputFile, "V:DrawProgressBar=True:Transformations=I;D;P;G:AnalysisType=Classification");
  
  TString path = "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_*.root";
  TChain* chain = new TChain("t");
  chain->Add(path);

  TObjArray *listOfFiles = chain->GetListOfFiles();
  TIter fileIter(listOfFiles);
  TFile *currentFile = 0;

  vector<TString> vFiles;

  while ( (currentFile = (TFile*)fileIter.Next()) ) 
    vFiles.push_back(currentFile->GetTitle());

  //vector<TString> vFiles = {"/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_1.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_2.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_3.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_4.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_5.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_6.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_7.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_8.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_9.root", "/hadoop/cms/store/user/smay/DeepIsolation/TTbar_DeepIso_v0.0.1_ptRelOrdered/merged_ntuple_10.root", };
  vector<TFile*> vFileSig;
  vector<TFile*> vFileBkg;
  vector<TTree*> vTreeSig;
  vector<TTree*> vTreeBkg;
  
  for (int i = 0; i < vFiles.size(); i++) {
    vFileSig.push_back(TFile::Open(vFiles[i]));
    vFileBkg.push_back(TFile::Open(vFiles[i]));
    vTreeSig.push_back((TTree*)vFileSig[i]->Get("t"));
    vTreeBkg.push_back((TTree*)vFileBkg[i]->Get("t"));
    factory->AddSignalTree(vTreeSig[i]);
    factory->AddBackgroundTree(vTreeBkg[i]);
  }

  Double_t signalWeight     = 1.0;
  Double_t backgroundWeight = 1.0;

  factory->AddVariable("lepton_eta", 'F');
  factory->AddVariable("lepton_phi", 'F');
  factory->AddVariable("lepton_pt", 'F');
  factory->AddVariable("lepton_relIso03EA", 'F');
  factory->AddVariable("lepton_chiso", 'F');
  factory->AddVariable("lepton_nhiso", 'F');
  factory->AddVariable("lepton_emiso", 'F');
  factory->AddVariable("lepton_ncorriso", 'F');
  factory->AddVariable("lepton_dxy", 'F');
  factory->AddVariable("lepton_dz", 'F');
  factory->AddVariable("lepton_ip3d", 'F');

  factory->AddVariable("lepton_nChargedPf", 'I');
  factory->AddVariable("lepton_nPhotonPf", 'I');
  factory->AddVariable("lepton_nNeutralHadPf", 'I');
  factory->AddVariable("nvtx", 'I');
  
  factory->AddVariable("pf_annuli_energy[0]", 'F');
  factory->AddVariable("pf_annuli_energy[1]", 'F');
  factory->AddVariable("pf_annuli_energy[2]", 'F');
  factory->AddVariable("pf_annuli_energy[3]", 'F');
  factory->AddVariable("pf_annuli_energy[4]", 'F');
  factory->AddVariable("pf_annuli_energy[5]", 'F');
  factory->AddVariable("pf_annuli_energy[6]", 'F');
  factory->AddVariable("pf_annuli_energy[7]", 'F');

  float nTrainF = nTrain;
  float nTrainSigF = nTrainF*0.875;
  float nTrainBkgF = nTrainF*0.125;

  int nTrainSig = (int) nTrainSigF;
  int nTrainBkg = (int) nTrainBkgF;

  int nTestSig = 87500;
  int nTestBkg = 12500;

  TString prepare_events = "nTrain_Signal=" + to_string(nTrainSig) + ":nTrain_Background=" + to_string(nTrainBkg) + ":nTest_Signal=" + to_string(nTestSig) + ":nTest_Background=" + to_string(nTestBkg) + ":SplitMode=Random:NormMode=NumEvents:!V";   

  factory->PrepareTrainingAndTestTree("lepton_isFromW==1&&lepton_flavor==1", "lepton_isFromW==0&&lepton_flavor==1", prepare_events);
  factory->SetSignalWeightExpression("1");
  factory->SetBackgroundWeightExpression("1");

  TString option = "!H:V:NTrees=1000:BoostType=Grad:Shrinkage=0.10:!UseBaggedGrad:nCuts=2000:MinNodeSize=0.1%:PruneStrength=5:PruneMethod=CostComplexity:MaxDepth=3:CreateMVAPdfs";

  factory->BookMethod(TMVA::Types::kBDT, "BDT", option);
  factory->TrainAllMethods();
  factory->TestAllMethods();
  factory->EvaluateAllMethods();

  outputFile->Close();
  std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
  std::cout << "==> TMVAClassification is done!" << std::endl;
  delete factory;
}
