#include <iostream>
#include <vector>
#include <set>

#include "TChain.h"
#include "TDirectory.h"
#include "TTreeCache.h"
#include "Math/VectorUtil.h"
#include "TVector2.h"
#include "TBenchmark.h"
#include "TLorentzVector.h"
#include "TH2.h"

#include "../CORE/CMS3.cc"
#include "../CORE/Base.cc"
#include "../CORE/OSSelections.cc"
//#include "../CORE/SSSelections.cc"
#include "../CORE/VVVSelections.cc"
#include "../CORE/ElectronSelections.cc"
#include "../CORE/IsolationTools.cc"
#include "../CORE/JetSelections.cc"
#include "../CORE/MuonSelections.cc"
#include "../CORE/IsoTrackVeto.cc"
#include "../CORE/PhotonSelections.cc"
#include "../CORE/TriggerSelections.cc"
#include "../CORE/VertexSelections.cc"
#include "../CORE/MCSelections.cc"
#include "../CORE/MetSelections.cc"
#include "../CORE/SimPa.cc"
#include "../CORE/Tools/jetcorr/FactorizedJetCorrector.h"
#include "../CORE/Tools/JetCorrector.cc"
#include "../CORE/Tools/jetcorr/JetCorrectionUncertainty.h"
//#include "../CORE/Tools/MT2/MT2.cc"
#include "../CORE/Tools/utils.cc"
#include "../CORE/Tools/goodrun.cc"
#include "../CORE/Tools/btagsf/BTagCalibrationStandalone.cc"

#include "JetSubstructureVariables.h"

using namespace std;
using namespace tas;

// "Good" mu/el id functions taken from Philip's old babymaker
//_________________________________________________________________________________________________
// Returns a vector of indices for good loose muons in CMS3.
std::vector<unsigned int> goodMuonIdx()
{
    // Loop over the muons and select good baseline muons.
    std::vector<unsigned int> good_muon_idx;
    for ( unsigned int imu = 0; imu < cms3.mus_p4().size(); ++imu )
    {
        if (!( isLooseMuonPOG( imu )                      )) continue;
        if (!( fabs(cms3.mus_p4()[imu].pt())     >  10    )) continue;
        if (!( fabs(cms3.mus_p4()[imu].eta())    <=  2.4  )) continue;
        if (!( fabs(cms3.mus_dxyPV()[imu])       <=  0.05 )) continue;
        if (!( fabs(cms3.mus_dzPV()[imu])        <=  0.1  )) continue;
        //if (!( muRelIso03EA( imu, 1 )            <   0.5  )) continue;
        good_muon_idx.push_back( imu );
    }
    return good_muon_idx;
}

//_________________________________________________________________________________________________
// Returns a vector of indices for good loose muons in CMS3.
std::vector<unsigned int> goodElecIdx()
{
    // Loop over the electrons and select good baseline electrons.
    std::vector<unsigned int> good_elec_idx;
    for ( unsigned int iel = 0; iel < cms3.els_p4().size(); ++iel )
    {
//      if (!( isTriggerSafenoIso_v1( iel )               )) continue; // If at some point we ever need it we can turn it on later.
        if (!( isVetoElectronPOGspring16noIso_v1( iel )   )) continue;
        if (!( fabs(cms3.els_p4()[iel].pt())     >  10    )) continue;
        if (!( fabs(cms3.els_p4()[iel].eta())    <=  2.5  )) continue;
        if (!( fabs(cms3.els_dxyPV()[iel])       <=  0.05 )) continue;
        if (!( fabs(cms3.els_dzPV()[iel])        <=  0.1  )) continue;
        //if (!( eleRelIso03EA( iel, 2 )           <   0.5  )) continue;
        good_elec_idx.push_back( iel );
    }
    return good_elec_idx;
}

void ScanReweights(TChain* chain, int max_events, TString filename) {
  TBenchmark *bmark = new TBenchmark();
  bmark->Start("benchmark");

  int nDuplicates = 0;
  int nEvents = chain->GetEntries();
  unsigned int nEventsChain = nEvents;
  cout << "Running on " << nEventsChain << " events" << endl;
  unsigned int nEventsTotal = 0;
  TObjArray *listOfFiles = chain->GetListOfFiles();
  TIter fileIter(listOfFiles);
  TFile *currentFile = 0;

  TFile* f1 = new TFile(filename, "RECREATE");
  f1->cd();

  double ptBins[] = {10, 20, 30, 40, 50, 60, 75, 100, 99999};
  int nPtBins = (sizeof(ptBins) / sizeof(ptBins[0])) - 1;

  double etaBins[] = {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0};
  int nEtaBins = (sizeof(etaBins) / sizeof(etaBins[0])) - 1;

  TH2D* hSig = new TH2D("hSig", "", nPtBins, ptBins, nEtaBins, etaBins);
  TH2D* hBkg = new TH2D("hBkg", "", nPtBins, ptBins, nEtaBins, etaBins);

  while ( (currentFile = (TFile*)fileIter.Next()) ) {
    cout << "running on file: " << currentFile->GetTitle() << endl;
    TString currentFileName(currentFile->GetTitle());

    // Get File Content
    TFile f( currentFile->GetTitle() );
    TTree *tree = (TTree*)f.Get("Events");
    TTreeCache::SetLearnEntries(10);
    tree->SetCacheSize(128*1024*1024);
    cms3.Init(tree);

    unsigned int nEventsToLoop = tree->GetEntriesFast();
    if (max_events > 0) nEventsToLoop = (unsigned int) max_events;

    //===============================
    // LOOP OVER EVENTS IN FILE
    //===============================
    for( unsigned int event = 0; event < nEventsToLoop; ++event) {
      // Get Event Content
      tree->LoadTree(event);
      cms3.GetEntry(event);
      ++nEventsTotal;

      // Progress
      CMS3::progress( nEventsTotal, nEventsChain );

      std::vector<unsigned int> good_muon_idx = goodMuonIdx();
      std::vector<unsigned int> good_elec_idx = goodElecIdx();

      int nGoodMuons = good_muon_idx.size();
      int nGoodElecs = good_elec_idx.size();

      std::vector<unsigned int> good_lep_idx;
      good_lep_idx.reserve(nGoodMuons + nGoodElecs);
      good_lep_idx.insert(good_lep_idx.end(), good_muon_idx.begin(), good_muon_idx.end());
      good_lep_idx.insert(good_lep_idx.end(), good_elec_idx.begin(), good_elec_idx.end());

      //////////////////////////
      // Loop through leptons //
      //////////////////////////

      for ( unsigned int lIdx = 0; lIdx < nGoodMuons + nGoodElecs; lIdx++ ) {
	unsigned int lepIdx = good_lep_idx[lIdx];
        bool isMu = lIdx < nGoodMuons;

        LorentzVector pLep = isMu ? cms3.mus_p4()[lepIdx] : cms3.els_p4()[lepIdx];

	// Lepton info
        int lepton_flavor = isMu ? 1 : 0;
        double lepton_eta = abs(pLep.eta());
        double lepton_pt  = pLep.pt() ;

        int pdgid = isMu ? 13 : 11;

        // Lepton truth
        int lepton_isFromW = isFromW(abs(pdgid), lepIdx);

        if (lepton_isFromW == 1)
          hSig->Fill(lepton_pt, lepton_eta);
        else if (lepton_isFromW == 0)
          hBkg->Fill(lepton_pt, lepton_eta);

      } // end loop on leptons

    } // end loop on events in file
  } // end loop on files

  cout << "Processed " << nEventsTotal << " events" << endl;
  bmark->Stop("benchmark");
  cout << endl;
  cout << nEventsTotal << " Events Processed" << endl;
  cout << "------------------------------" << endl;
  cout << "CPU  Time:   " << Form( "%.01f s", bmark->GetCpuTime("benchmark")  ) << endl;
  cout << "Real Time:   " << Form( "%.01f s", bmark->GetRealTime("benchmark") ) << endl;
  cout << endl;

  TH2D* hProb = (TH2D*)hSig->Clone();
  hProb->Divide(hBkg);
  hProb->Scale(1/hProb->GetMaximum());

  f1->Write();
  f1->Close();

  return;
}

