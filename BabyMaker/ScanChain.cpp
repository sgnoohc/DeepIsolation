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

#include "ScanChain.h"

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
        if (!( muRelIso03EA( imu, 1 )            <   0.5  )) continue;
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
        if (!( eleRelIso03EA( iel, 2 )           <   0.5  )) continue;
        good_elec_idx.push_back( iel );
    }
    return good_elec_idx;
}



void BabyMaker::ScanChain(TChain* chain, std::string baby_name, int max_events){

  // Benchmark
  TBenchmark *bmark = new TBenchmark();
  bmark->Start("benchmark");

  MakeBabyNtuple( Form("%s.root", baby_name.c_str()) );

  int nDuplicates = 0;
  int nEvents = chain->GetEntries();
  unsigned int nEventsChain = nEvents;
  cout << "Running on " << nEventsChain << " events" << endl;
  unsigned int nEventsTotal = 0;
  TObjArray *listOfFiles = chain->GetListOfFiles();
  TIter fileIter(listOfFiles);
  TFile *currentFile = 0;

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

      InitBabyNtuple();

      run    = cms3.evt_run();
      lumi   = cms3.evt_lumiBlock();
      evt    = cms3.evt_event();

      nvtx = 0;
      for ( unsigned int ivtx = 0; ivtx < cms3.evt_nvtxs(); ivtx++ )
	  if ( isGoodVertex( ivtx ) ) nvtx++;
      
      // Identify good leptons
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
  
        lepton_flavor = isMu ? 1 : 0;
      
        lepton_eta = pLep.eta();
        lepton_phi = pLep.phi();
        lepton_pt  = pLep.pt() ;

        int pdgid = isMu ? 13 : 11;

        lepton_isFromW = isFromW(abs(pdgid), lepIdx);
        lepton_isFromB = isFromB(abs(pdgid), lepIdx);
	lepton_isFromC = isFromC(abs(pdgid), lepIdx);
	lepton_isFromL = isFromLight(abs(pdgid), lepIdx);
	lepton_isFromLF = isFromLightFake(abs(pdgid), lepIdx);

        lepton_relIso03EA = isMu ? muRelIso03EA(lepIdx, 1) : eleRelIso03EA(lepIdx, 2);
        lepton_chiso = isMu ? cms3.mus_isoR03_pf_ChargedHadronPt()[lepIdx] : cms3.els_pfChargedHadronIso()[lepIdx];
        lepton_nhiso = isMu ? cms3.mus_isoR03_pf_NeutralHadronEt()[lepIdx] : cms3.els_pfNeutralHadronIso()[lepIdx];
        lepton_emiso = isMu ? cms3.mus_isoR03_pf_PhotonEt()[lepIdx] : cms3.els_pfPhotonIso()[lepIdx];
        lepton_ncorriso = lepton_nhiso + lepton_emiso - evt_fixgridfastjet_all_rho() * (isMu ? muEA03(lepIdx, 1) : elEA03(lepIdx, 2));

        lepton_dxy  = isMu ? cms3.mus_dxyPV()[lepIdx] : cms3.els_dxyPV()[lepIdx];
        lepton_dz   = isMu ? cms3.mus_dzPV()[lepIdx] : cms3.els_dzPV()[lepIdx];
        lepton_ip3d = isMu ? cms3.mus_ip3d()[lepIdx] : cms3.els_ip3d()[lepIdx];

        FillBabyNtuple();

        } // end lepton loop

      } // end loop on events in file
    delete tree;
    f.Close();
  } // end loop on files

  cout << "Processed " << nEventsTotal << " events" << endl;
  if ( nEventsChain != nEventsTotal ) {
    std::cout << "ERROR: number of events from files is not equal to total number of events" << std::endl;
  }

  CloseBabyNtuple();

  bmark->Stop("benchmark");
  cout << endl;
  cout << nEventsTotal << " Events Processed" << endl;
  cout << "------------------------------" << endl;
  cout << "CPU  Time:	" << Form( "%.01f s", bmark->GetCpuTime("benchmark")  ) << endl;
  cout << "Real Time:	" << Form( "%.01f s", bmark->GetRealTime("benchmark") ) << endl;
  cout << endl;

  return;
}

void BabyMaker::MakeBabyNtuple(const char *BabyFilename){
  BabyFile_ = new TFile(Form("%s", BabyFilename), "RECREATE");
  BabyFile_->cd();
  BabyTree_ = new TTree("t", "A Baby Ntuple");

  BabyTree_->Branch("run"   , &run    );
  BabyTree_->Branch("lumi"  , &lumi   );
  BabyTree_->Branch("evt"   , &evt    );

  BabyTree_->Branch("nvtx"   , &nvtx    );

  BabyTree_->Branch("lepton_flavor"   , &lepton_flavor    );
  BabyTree_->Branch("lepton_isFromW"   , &lepton_isFromW    );
  BabyTree_->Branch("lepton_isFromB"   , &lepton_isFromB    );
  BabyTree_->Branch("lepton_isFromC"   , &lepton_isFromC    );
  BabyTree_->Branch("lepton_isFromL"   , &lepton_isFromL    );
  BabyTree_->Branch("lepton_isFromLF"   , &lepton_isFromLF    ); 

  BabyTree_->Branch("lepton_eta"   , &lepton_eta    );
  BabyTree_->Branch("lepton_phi"   , &lepton_phi    );
  BabyTree_->Branch("lepton_pt"   , &lepton_pt    );
  BabyTree_->Branch("lepton_relIso03EA"   , &lepton_relIso03EA    );
  BabyTree_->Branch("lepton_chiso"   , &lepton_chiso    );
  BabyTree_->Branch("lepton_nhiso"   , &lepton_nhiso    );
  BabyTree_->Branch("lepton_emiso"   , &lepton_emiso    );
  BabyTree_->Branch("lepton_ncorriso"   , &lepton_ncorriso    );
  BabyTree_->Branch("lepton_dxy"   , &lepton_dxy    );
  BabyTree_->Branch("lepton_dz"   , &lepton_dz    );
  BabyTree_->Branch("lepton_ip3d"   , &lepton_ip3d    );

  return;
}

void BabyMaker::InitBabyNtuple () {
  run    = -999;
  lumi   = -999;
  evt    = -1;

  nvtx   = -999;

  //pf_charged_pt.clear()

  return;
}

void BabyMaker::FillBabyNtuple(){
  BabyTree_->Fill();
  return;
}

void BabyMaker::CloseBabyNtuple(){
  BabyFile_->cd();
  BabyTree_->Write();
  //h_neventsinfile->Write();
  BabyFile_->Close();
  return;
}

