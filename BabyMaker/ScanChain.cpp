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
#include "JetSubstructureVariables.h"

using namespace std;
using namespace tas;

const double coneSize = 0.5;
const int nAnnuli = 8;
const double coneSizeAnnuli = 1.0;

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

//_________________________________________________________________________________________________
bool sortByValue(const std::pair<int,float>& pair1, const std::pair<int,float>& pair2 ) {
  return pair1.second > pair2.second;
}

//_________________________________________________________________________________________________
//const double pi = 3.1415926536; // Already defined in fastjet with higher precision
double alpha(const LorentzVector p1, const LorentzVector p2) {
  double phi = p2.phi() - p1.phi();
  if (abs(phi) > pi)
    phi = p2.phi() + p1.phi();
  double eta = p2.eta() - p1.eta();
  return TMath::ATan2(eta, phi);
}

//_________________________________________________________________________________________________
void BabyMaker::ScanChain(TChain* chain, std::string baby_name, int max_events){

  // Benchmark
  TBenchmark *bmark = new TBenchmark();
  bmark->Start("benchmark");

  MakeBabyNtuple( Form("%s.root", baby_name.c_str()) );

  int nDuplicates = 0;
  int nEvents = chain->GetEntries();
  int nLeptons = 0;
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
  
        // Lepton info
        lepton_flavor = isMu ? 1 : 0;      
        lepton_eta = pLep.eta();
        lepton_phi = pLep.phi();
        lepton_pt  = pLep.pt() ;

        int pdgid = isMu ? 13 : 11;

        // Lepton truth
        lepton_isFromW = isFromW(abs(pdgid), lepIdx);
        lepton_isFromB = isFromB(abs(pdgid), lepIdx);
        lepton_isFromC = isFromC(abs(pdgid), lepIdx);
        lepton_isFromL = isFromLight(abs(pdgid), lepIdx);
        lepton_isFromLF = isFromLightFake(abs(pdgid), lepIdx);

        // Lepton isolation vars
        lepton_relIso03EA = isMu ? muRelIso03EA(lepIdx, 1) : eleRelIso03EA(lepIdx, 2);
        lepton_chiso = isMu ? cms3.mus_isoR03_pf_ChargedHadronPt()[lepIdx] : cms3.els_pfChargedHadronIso()[lepIdx];
        lepton_nhiso = isMu ? cms3.mus_isoR03_pf_NeutralHadronEt()[lepIdx] : cms3.els_pfNeutralHadronIso()[lepIdx];
        lepton_emiso = isMu ? cms3.mus_isoR03_pf_PhotonEt()[lepIdx] : cms3.els_pfPhotonIso()[lepIdx];
        lepton_ncorriso = lepton_nhiso + lepton_emiso - evt_fixgridfastjet_all_rho() * (isMu ? muEA03(lepIdx, 1) : elEA03(lepIdx, 2));

        // Impact parameter
        lepton_dxy  = isMu ? cms3.mus_dxyPV()[lepIdx] : cms3.els_dxyPV()[lepIdx];
        lepton_dz   = isMu ? cms3.mus_dzPV()[lepIdx] : cms3.els_dzPV()[lepIdx];
        lepton_ip3d = isMu ? cms3.mus_ip3d()[lepIdx] : cms3.els_ip3d()[lepIdx];

        // substructure variables
        int ijet = matchedJetIdx(pLep);
        substr_ptrel = getPtRel(pdgid, lepIdx, true, 2);
        substr_jetpt = ijet < 0 ? pLep.pt() : cms3.pfjets_p4()[ijet].pt();
        std::tie(substr_subjet_pt, substr_subjet_eta, substr_subjet_phi, substr_subjet_e, substr_nsubjets, substr_dijs, substr_dRs, substr_minkts) = get_dij_components(pLep);
        TLorentzVector tlv_subjet;
        tlv_subjet.SetPtEtaPhiE(substr_subjet_pt, substr_subjet_eta, substr_subjet_phi, substr_subjet_e);
        LorentzVector lv_subjet;
        lv_subjet.SetPxPyPzE(tlv_subjet.Px(), tlv_subjet.Py(), tlv_subjet.Pz(), tlv_subjet.E());
        substr_subjet_dr = ROOT::Math::VectorUtil::DeltaR(lv_subjet, pLep);
        substr_sumdij = 0;
        substr_maxdij = 0;
        substr_mindij = 9999999;
        substr_ndij = substr_dijs.size();
        substr_njet = get_nmatched_jets(pLep);
        for (auto& dij : substr_dijs)
        {
          if (dij > substr_maxdij) substr_maxdij = dij;
          if (dij < substr_mindij) substr_mindij = dij;
          substr_sumdij += dij;
        }
        std::tie(substr_pf_pt, substr_pf_eta, substr_pf_phi, substr_pf_dr, substr_pf_type, substr_pf_id) = get_pf_cands(pLep);
        vector<LorentzVector> reclsjs = get_subjets(pLep, antikt_algorithm, 0.6);
        substr_nreclsj = 0;
        substr_reclsj_dr.clear();
        substr_reclsj_pt.clear();
        substr_reclsj_eta.clear();
        substr_reclsj_phi.clear();
        substr_reclsj_e.clear();
        substr_reclsj_m.clear();
        substr_nreclsj = reclsjs.size();
        for (auto& jet : reclsjs)
        {
            substr_reclsj_pt.push_back(jet.pt());
            substr_reclsj_eta.push_back(jet.eta());
            substr_reclsj_phi.push_back(jet.phi());
            substr_reclsj_e.push_back(jet.e());
            substr_reclsj_m.push_back(jet.mass());
            substr_reclsj_dr.push_back(ROOT::Math::VectorUtil::DeltaR(pLep, jet));
        }

        ////////////////////////////////
        // Loop through pf candidates //
        ////////////////////////////////

        lepton_nChargedPf    = 0;
        lepton_nPhotonPf     = 0;
        lepton_nNeutralHadPf = 0;

        std::vector<Float_t> unordered_pf_charged_pt                     ;
        std::vector<Float_t> unordered_pf_charged_dR                     ;
        std::vector<Float_t> unordered_pf_charged_alpha                  ;
        std::vector<Float_t> unordered_pf_charged_ptRel                  ;
        std::vector<Float_t> unordered_pf_charged_puppiWeight            ;
        std::vector<Int_t>   unordered_pf_charged_fromPV                 ;
        std::vector<Int_t>   unordered_pf_charged_pvAssociationQuality   ;

        std::vector<Float_t> unordered_pf_photon_pt          ;
        std::vector<Float_t> unordered_pf_photon_dR          ;
        std::vector<Float_t> unordered_pf_photon_alpha       ;
        std::vector<Float_t> unordered_pf_photon_ptRel       ;
        std::vector<Float_t> unordered_pf_photon_puppiWeight ;

        std::vector<Float_t> unordered_pf_neutralHad_pt          ;
        std::vector<Float_t> unordered_pf_neutralHad_dR          ;
        std::vector<Float_t> unordered_pf_neutralHad_alpha       ;
        std::vector<Float_t> unordered_pf_neutralHad_ptRel       ;
        std::vector<Float_t> unordered_pf_neutralHad_puppiWeight ;

        std::vector<std::pair<int, float> > charged_pt_ordering;
        std::vector<std::pair<int, float> > photon_pt_ordering;
        std::vector<std::pair<int, float> > neutralHad_pt_ordering;

        for (int i = 0; i < nAnnuli; i++)
          pf_annuli_energy.push_back(0);

        TH1D* hR = new TH1D("hR", "", nAnnuli, 0.0, coneSizeAnnuli);

        // Begin pf cand loop
        for ( unsigned int pIdx = 0; pIdx < cms3.pfcands_p4().size(); pIdx++ ) {
          LorentzVector pCand = cms3.pfcands_p4()[pIdx];
          double dR = DeltaR(pLep, pCand);
          if (dR < coneSizeAnnuli) {
            int idx = (hR->FindBin(dR))-1;
            pf_annuli_energy[idx] += pCand.pt();
          }

          if (dR > coneSize) continue;

          int pf_pdg_id = cms3.pfcands_particleId()[pIdx];
          int pf_charge = cms3.pfcands_charge()[pIdx];
    
          // Check if pf cand is lepton itself
          if (abs(pf_pdg_id) == (isMu ? 13 : 11 ) && dR < 0.05) continue;

          // Identify charged/photon/neutral hadron
          int candIdx = -1; // 0 = charged, 1 = photon, 2 = neutral hadron
          if (abs(pf_charge) > 0) candIdx = 0;
          else if (abs(pf_pdg_id) == 22) candIdx = 1;
          else candIdx = 2;

          bool orderByPt = true; // true = order by pT, false = order by pTRel

          if (candIdx == 0) { // charged
            if (orderByPt)
              charged_pt_ordering.push_back(std::pair<int, float>(lepton_nChargedPf, pCand.pt()));
            else
              charged_pt_ordering.push_back(std::pair<int, float>(lepton_nChargedPf, ptRel(pCand, pLep, false)));
            lepton_nChargedPf++;

            unordered_pf_charged_pt.push_back(pCand.pt()/pLep.pt());
            unordered_pf_charged_dR.push_back(DeltaR(pLep, pCand));
            unordered_pf_charged_alpha.push_back(alpha(pLep, pCand));
            unordered_pf_charged_ptRel.push_back(ptRel(pCand, pLep, false));
            unordered_pf_charged_puppiWeight.push_back(cms3.pfcands_puppiWeight()[pIdx]);

            unordered_pf_charged_fromPV.push_back(cms3.pfcands_fromPV()[pIdx]);
            unordered_pf_charged_pvAssociationQuality.push_back(cms3.pfcands_pvAssociationQuality()[pIdx]);
          }

          else if (candIdx == 1) { // photons
            if (orderByPt)
              photon_pt_ordering.push_back(std::pair<int, float>(lepton_nPhotonPf, pCand.pt()));
            else
              photon_pt_ordering.push_back(std::pair<int, float>(lepton_nPhotonPf, ptRel(pCand, pLep, false)));
            lepton_nPhotonPf++;

            unordered_pf_photon_pt.push_back(pCand.pt()/pLep.pt());
            unordered_pf_photon_dR.push_back(DeltaR(pLep, pCand));
            unordered_pf_photon_alpha.push_back(alpha(pLep, pCand));
            unordered_pf_photon_ptRel.push_back(ptRel(pCand, pLep, false));
            unordered_pf_photon_puppiWeight.push_back(cms3.pfcands_puppiWeight()[pIdx]);
          }

          else if (candIdx == 2) { // neutral hadrons
            if (orderByPt)
              neutralHad_pt_ordering.push_back(std::pair<int, float>(lepton_nNeutralHadPf, pCand.pt()));
            else
              neutralHad_pt_ordering.push_back(std::pair<int, float>(lepton_nNeutralHadPf, ptRel(pCand, pLep, false)));
            lepton_nNeutralHadPf++;

            unordered_pf_neutralHad_pt.push_back(pCand.pt()/pLep.pt());
            unordered_pf_neutralHad_dR.push_back(DeltaR(pLep, pCand));
            unordered_pf_neutralHad_alpha.push_back(alpha(pLep, pCand));
            unordered_pf_neutralHad_ptRel.push_back(ptRel(pCand, pLep, false));
            unordered_pf_neutralHad_puppiWeight.push_back(cms3.pfcands_puppiWeight()[pIdx]);
          }

        } // end pf cand loop 
        delete hR;

        // Sort charged pf cands
        std::sort(charged_pt_ordering.begin(), charged_pt_ordering.end(), sortByValue);
        for (std::vector<std::pair<int, float> >::iterator it = charged_pt_ordering.begin(); it != charged_pt_ordering.end(); ++it) {
          pf_charged_pt.push_back(unordered_pf_charged_pt.at(it->first));
          pf_charged_dR.push_back(unordered_pf_charged_dR.at(it->first));
          pf_charged_alpha.push_back(unordered_pf_charged_alpha.at(it->first));
          pf_charged_ptRel.push_back(unordered_pf_charged_ptRel.at(it->first));
          pf_charged_puppiWeight.push_back(unordered_pf_charged_puppiWeight.at(it->first));

          pf_charged_fromPV.push_back(unordered_pf_charged_fromPV.at(it->first));
          pf_charged_pvAssociationQuality.push_back(unordered_pf_charged_pvAssociationQuality.at(it->first));
        }
       
        // Sort photon cands
        std::sort(photon_pt_ordering.begin(), photon_pt_ordering.end(), sortByValue);
        for (std::vector<std::pair<int, float> >::iterator it = photon_pt_ordering.begin(); it != photon_pt_ordering.end(); ++it) {
          pf_photon_pt.push_back(unordered_pf_photon_pt.at(it->first));
          pf_photon_dR.push_back(unordered_pf_photon_dR.at(it->first));
          pf_photon_alpha.push_back(unordered_pf_photon_alpha.at(it->first));
          pf_photon_ptRel.push_back(unordered_pf_photon_ptRel.at(it->first));
          pf_photon_puppiWeight.push_back(unordered_pf_photon_puppiWeight.at(it->first));
        }

        // Sort neutral hads
        std::sort(neutralHad_pt_ordering.begin(), neutralHad_pt_ordering.end(), sortByValue);
        for (std::vector<std::pair<int, float> >::iterator it = neutralHad_pt_ordering.begin(); it != neutralHad_pt_ordering.end(); ++it) {
          pf_neutralHad_pt.push_back(unordered_pf_neutralHad_pt.at(it->first));
          pf_neutralHad_dR.push_back(unordered_pf_neutralHad_dR.at(it->first));
          pf_neutralHad_alpha.push_back(unordered_pf_neutralHad_alpha.at(it->first));
          pf_neutralHad_ptRel.push_back(unordered_pf_neutralHad_ptRel.at(it->first));
          pf_neutralHad_puppiWeight.push_back(unordered_pf_neutralHad_puppiWeight.at(it->first));
        }

        FillBabyNtuple();
        nLeptons++;

        pf_charged_pt.clear();
        pf_charged_dR.clear();
        pf_charged_alpha.clear();
        pf_charged_ptRel.clear();
        pf_charged_puppiWeight.clear();
        pf_charged_fromPV.clear();
        pf_charged_pvAssociationQuality.clear();

        pf_photon_pt.clear();
        pf_photon_dR.clear();
        pf_photon_alpha.clear();
        pf_photon_ptRel.clear();
        pf_photon_puppiWeight.clear();

        pf_neutralHad_pt.clear();
        pf_neutralHad_dR.clear();
        pf_neutralHad_alpha.clear();
        pf_neutralHad_ptRel.clear();
        pf_neutralHad_puppiWeight.clear();

	pf_annuli_energy.clear();

      } // end lepton loop

    } // end loop on events in file
    delete tree;
    f.Close();
  } // end loop on files

  cout << "Processed " << nEventsTotal << " events" << endl;
  cout << "Found " << nLeptons << " leptons" << endl;

  CloseBabyNtuple();

  bmark->Stop("benchmark");
  cout << endl;
  cout << nEventsTotal << " Events Processed" << endl;
  cout << "------------------------------" << endl;
  cout << "CPU  Time:   " << Form( "%.01f s", bmark->GetCpuTime("benchmark")  ) << endl;
  cout << "Real Time:   " << Form( "%.01f s", bmark->GetRealTime("benchmark") ) << endl;
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

  BabyTree_->Branch("substr_ptrel"    , &substr_ptrel     );
  BabyTree_->Branch("substr_jetpt"    , &substr_jetpt     );
  BabyTree_->Branch("substr_dijs"     , &substr_dijs      );
  BabyTree_->Branch("substr_dRs"      , &substr_dRs       );
  BabyTree_->Branch("substr_minkts"   , &substr_minkts    );
  BabyTree_->Branch("substr_sumdij"   , &substr_sumdij    );
  BabyTree_->Branch("substr_maxdij"   , &substr_maxdij    );
  BabyTree_->Branch("substr_mindij"   , &substr_mindij    );
  BabyTree_->Branch("substr_ndij"     , &substr_ndij      );
  BabyTree_->Branch("substr_njet"     , &substr_njet      );
  BabyTree_->Branch("substr_pf_pt"    , &substr_pf_pt     );
  BabyTree_->Branch("substr_pf_eta"   , &substr_pf_eta    );
  BabyTree_->Branch("substr_pf_phi"   , &substr_pf_phi    );
  BabyTree_->Branch("substr_pf_dr"    , &substr_pf_dr     );
  BabyTree_->Branch("substr_pf_type"  , &substr_pf_type   );
  BabyTree_->Branch("substr_pf_id"      , &substr_pf_id   );
  BabyTree_->Branch("substr_subjet_pt"  , &substr_subjet_pt  );
  BabyTree_->Branch("substr_subjet_eta" , &substr_subjet_eta );
  BabyTree_->Branch("substr_subjet_phi" , &substr_subjet_phi );
  BabyTree_->Branch("substr_subjet_dr"  , &substr_subjet_dr  );
  BabyTree_->Branch("substr_nsubjets"   , &substr_nsubjets   );
  BabyTree_->Branch("substr_nreclsj"  , &substr_nreclsj  );
  BabyTree_->Branch("substr_reclsj_dr"  , &substr_reclsj_dr  );
  BabyTree_->Branch("substr_reclsj_pt"  , &substr_reclsj_pt  );
  BabyTree_->Branch("substr_reclsj_eta"  , &substr_reclsj_eta  );
  BabyTree_->Branch("substr_reclsj_phi"  , &substr_reclsj_phi  );
  BabyTree_->Branch("substr_reclsj_e"  , &substr_reclsj_e  );
  BabyTree_->Branch("substr_reclsj_m"  , &substr_reclsj_m  );

  BabyTree_->Branch("lepton_nChargedPf", &lepton_nChargedPf);
  BabyTree_->Branch("lepton_nPhotonPf", &lepton_nPhotonPf);
  BabyTree_->Branch("lepton_nNeutralHadPf", &lepton_nNeutralHadPf);

  BabyTree_->Branch("pf_charged_pt"   , &pf_charged_pt    );
  BabyTree_->Branch("pf_charged_dR"   , &pf_charged_dR    );
  BabyTree_->Branch("pf_charged_alpha"   , &pf_charged_alpha    );
  BabyTree_->Branch("pf_charged_ptRel"   , &pf_charged_ptRel    );
  BabyTree_->Branch("pf_charged_puppiWeight"   , &pf_charged_puppiWeight    );
  BabyTree_->Branch("pf_charged_fromPV"   , &pf_charged_fromPV    );
  BabyTree_->Branch("pf_charged_pvAssociationQuality"   , &pf_charged_pvAssociationQuality    );

  BabyTree_->Branch("pf_photon_pt"   , &pf_photon_pt    );
  BabyTree_->Branch("pf_photon_dR"   , &pf_photon_dR    );
  BabyTree_->Branch("pf_photon_alpha"   , &pf_photon_alpha    );
  BabyTree_->Branch("pf_photon_ptRel"   , &pf_photon_ptRel    );
  BabyTree_->Branch("pf_photon_puppiWeight"   , &pf_photon_puppiWeight    );

  BabyTree_->Branch("pf_neutralHad_pt"   , &pf_neutralHad_pt    );
  BabyTree_->Branch("pf_neutralHad_dR"   , &pf_neutralHad_dR    );
  BabyTree_->Branch("pf_neutralHad_alpha"   , &pf_neutralHad_alpha    );
  BabyTree_->Branch("pf_neutralHad_ptRel"   , &pf_neutralHad_ptRel    );
  BabyTree_->Branch("pf_neutralHad_puppiWeight"   , &pf_neutralHad_puppiWeight    );

  BabyTree_->Branch("pf_annuli_energy" , &pf_annuli_energy);

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

