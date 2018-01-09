#ifndef ScanChain_h
#define ScanChain_h

#include <string>
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TH2.h"
#include "TString.h"
#include "Math/LorentzVector.h"
#include "Math/GenVector/LorentzVector.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;

class BabyMaker {
  public:
    BabyMaker() {};
    ~BabyMaker() {
      if (BabyFile_) delete BabyFile_;
      if (BabyTree_) delete BabyTree_;
    }
    void ScanChain(TChain*, std::string = "testSample", int max_events = -1);
    void MakeBabyNtuple(const char *);
    void InitBabyNtuple();
    void FillBabyNtuple();
    void CloseBabyNtuple();

  private:
    TFile *BabyFile_;
    TTree *BabyTree_;
    
    Int_t           run;
    Int_t           lumi;
    ULong64_t       evt;

    Int_t   lepton_flavor     ;
    Int_t   lepton_isFromW    ;
    Int_t   lepton_isFromB    ;
    Int_t   lepton_isFromC    ;
    Int_t   lepton_isFromL    ;
    Int_t   lepton_isFromLF   ;

    Int_t   nvtx	         ;
    Float_t lepton_eta           ;
    Float_t lepton_phi           ;
    Float_t lepton_pt            ;
    Float_t lepton_relIso03EA    ;
    Float_t lepton_chiso         ;
    Float_t lepton_nhiso         ;
    Float_t lepton_emiso         ;
    Float_t lepton_ncorriso      ;
    Float_t lepton_dxy           ;
    Float_t lepton_dz            ;
    Float_t lepton_ip3d          ;

    Int_t   lepton_nChargedPf    ;
    Int_t   lepton_nPhotonPf     ;
    Int_t   lepton_nNeutralHadPf ;

    std::vector<Float_t> pf_charged_pt			   ;
    std::vector<Float_t> pf_charged_dR			   ;
    std::vector<Float_t> pf_charged_ptRel		   ;
    std::vector<Float_t> pf_charged_puppiWeight 	   ;
    std::vector<Int_t>   pf_charged_fromPV		   ;
    std::vector<Int_t>   pf_charged_pvAssociationQuality   ;

    std::vector<Float_t> pf_photon_pt          ;
    std::vector<Float_t> pf_photon_dR          ;
    std::vector<Float_t> pf_photon_ptRel       ;
    std::vector<Float_t> pf_photon_puppiWeight ;

    std::vector<Float_t> pf_neutralHad_pt          ;
    std::vector<Float_t> pf_neutralHad_dR          ;
    std::vector<Float_t> pf_neutralHad_ptRel       ;
    std::vector<Float_t> pf_neutralHad_puppiWeight ;
};

#endif
