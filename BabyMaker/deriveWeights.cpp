#include "ScanReweights.cpp"

const TString savename = "weights/weights_PtEta.root";

using namespace std;

int main(int argc, char* argv[]) {
  TString path = "/hadoop/cms/store/group/snt/run2_moriond17/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/V08-00-16/merged_ntuple_1.root";

  TChain *ch = new TChain("Events");
  ch->Add(path);

  ScanReweights(ch, -1, savename);
  return 0;
}
