//
#ifndef JETSUBSTRUCTUREVARIABLES_H
#define JETSUBSTRUCTUREVARIABLES_H

#include <tuple>
using namespace std;

#include "fastjet/ClusterSequence.hh"
using namespace fastjet;

//##############################################################################
int matchedJetIdx(LorentzVector lep_p4)
{
    int index = -1;
    float mindr = 9999;
    for (unsigned int iJet = 0; iJet < cms3.pfjets_p4().size(); iJet++)
    {
        LorentzVector jet_p4 = cms3.pfjets_p4()[iJet];
        float tmpdr = ROOT::Math::VectorUtil::DeltaR(jet_p4, lep_p4);
        if (tmpdr < mindr)
        {
            mindr = tmpdr;
            index = iJet;
        }
    }
    return index;
}

//##############################################################################
int get_pf_type(int pIdx)
{
    int pf_pdg_id = cms3.pfcands_particleId()[pIdx];
    int pf_charge = cms3.pfcands_charge()[pIdx];
    // Check if pf cand is lepton itself
    int candIdx = 0;
    if (abs(pf_pdg_id) == 13 || abs(pf_pdg_id) == 11) candIdx = -1;
    else if (abs(pf_charge) > 0) candIdx = 0;
    else if (abs(pf_pdg_id) == 22) candIdx = 1;
    else candIdx = 2;
    return candIdx;
}

//#############################################################################
// Aggregate the dij distance parameters
void recursive_get_dijs(ClusterSequence cs, int jetidx, int depth, PseudoJet jet, vector<float>& dijs, vector<float>& dRs, vector<float>& minkts)
{
    if (depth == 0)
    {
        if (sorted_by_pt(cs.inclusive_jets()).size() == 0)
            return;
        PseudoJet j = sorted_by_pt(cs.inclusive_jets())[jetidx];
        dijs.clear();
        dRs.clear();
        minkts.clear();
        recursive_get_dijs(cs, jetidx, depth + 2, j, dijs, dRs, minkts);
    }
    else
    {
        if (jet.pieces().size() == 2)
        {
//            float dij = jet.pieces()[0].kt_distance(jet.pieces()[1]);
            float dR = jet.pieces()[0].delta_R(jet.pieces()[1]);
            float minkt = min(jet.pieces()[0].pt(), jet.pieces()[1].pt());
            float maxkt = max(jet.pieces()[0].pt(), jet.pieces()[1].pt());
//            float dij = dR * dR / maxkt / maxkt;
            float dij = dR * dR * minkt * minkt;
            dijs.push_back(dij);
            dRs.push_back(dR);
            minkts.push_back(minkt);
            for (auto& p : jet.pieces())
            {
                recursive_get_dijs(cs, jetidx, depth + 2, p, dijs, dRs, minkts);
            }
        }
        if (jet.pieces().size() == 0)
        {
            return;
        }
    }
}

//#############################################################################
// From a list of 4-vectors called "seeds",
// return a list of "jets" clustered based on "algo"
std::tuple<float, float, float, float, int, vector<float>, vector<float>, vector<float>> compute_dijs(vector<LorentzVector> seeds, JetAlgorithm algo=kt_algorithm)
{
    vector<PseudoJet> particles;

    for (LorentzVector& lv : seeds)
        particles.push_back(PseudoJet(lv.px(), lv.py(), lv.pz(), lv.e()));

    // choose a jet definition
    JetDefinition jet_def(algo, 0.4);

    // run the clustering, extract the jets
    ClusterSequence cs(particles, jet_def);
    vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets());

    vector<float> dijs;
    vector<float> dRs;
    vector<float> minkts;
    if (jets.size() == 0)
        return make_tuple(0, 0, 0, 0, 0, dijs, dRs, minkts);

    recursive_get_dijs(cs, 0, 0, PseudoJet(), dijs, dRs, minkts);

    return make_tuple(jets[0].pt(), jets[0].eta(), jets[0].phi(), jets[0].e(), jets.size(), dijs, dRs, minkts);
}

//##############################################################################
std::tuple<float, float, float, float, int, vector<float>, vector<float>, vector<float>> get_dij_components(LorentzVector lep_p4, JetAlgorithm algo=kt_algorithm, bool removecore=false, bool removenonPV=true)
{

    int iJet = matchedJetIdx(lep_p4);
    if (iJet < 0)
        return make_tuple(0, 0, 0, 0, 0, vector<float>(), vector<float>(), vector<float>());

    vector<LorentzVector> seeds;
    for (unsigned int ipf = 0; ipf < cms3.pfjets_pfcandIndicies()[iJet].size(); ++ipf)
    {
        int idx = cms3.pfjets_pfcandIndicies()[iJet][ipf];

        bool iscore = abs(lep_p4.pt() - cms3.pfcands_p4()[idx].pt()) < 0.001;

        if (removecore and iscore)
            continue;

        if (removenonPV and !iscore)
            if (get_pf_type(idx) < 0)
                continue;

        seeds.push_back(cms3.pfcands_p4()[idx]);
    }
    return compute_dijs(seeds, algo);

//    bool matched = false;
//    for (unsigned int iJet = 0; iJet < cms3.pfjets_p4().size(); iJet++)
//    {
//        LorentzVector jet_p4 = cms3.pfjets_p4()[iJet];
//        if (ROOT::Math::VectorUtil::DeltaR(jet_p4, lep_p4) < 0.4)
//        {
//            vector<LorentzVector> seeds;
//            for (unsigned int ipf = 0; ipf < cms3.pfjets_pfcandIndicies()[iJet].size(); ++ipf)
//            {
//                int idx = cms3.pfjets_pfcandIndicies()[iJet][ipf];
//                if (removecore)
//                    if (abs(lep_p4.pt() - cms3.pfcands_p4()[idx].pt()) < 0.001)
//                        continue;
//                if (removenonPV)
//                    if (get_pf_type(idx) < 0)
//                        continue;
//
//                seeds.push_back(cms3.pfcands_p4()[idx]);
//            }
//            vector<float> dijs;
//            vector<float> dRs;
//            vector<float> minkts;
//            return compute_dijs(seeds, algo);
//        }
//    }
//    return make_tuple(vector<float>(), vector<float>(), vector<float>());
}

//##############################################################################
int get_nmatched_jets(LorentzVector lep_p4)
{
    int nmatched = 0;
    for (unsigned int iJet = 0; iJet < cms3.pfjets_p4().size(); iJet++)
    {
        LorentzVector jet_p4 = cms3.pfjets_p4()[iJet];
        if (ROOT::Math::VectorUtil::DeltaR(jet_p4, lep_p4) < 0.4)
            nmatched++;
    }
    return nmatched;
}

//##############################################################################
std::tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<int>, vector<int>> get_pf_cands(LorentzVector lep_p4)
{
    vector<float> pf_pt;
    vector<float> pf_eta;
    vector<float> pf_phi;
    vector<float> pf_dr;
    vector<int> pf_type;
    vector<int> pf_id;
    for (unsigned int iJet = 0; iJet < cms3.pfjets_p4().size(); iJet++)
    {
        LorentzVector jet_p4 = cms3.pfjets_p4()[iJet];
        if (ROOT::Math::VectorUtil::DeltaR(jet_p4, lep_p4) < 0.4)
        {
            vector<LorentzVector> seeds;
            for (unsigned int ipf = 0; ipf < cms3.pfjets_pfcandIndicies()[iJet].size(); ++ipf)
            {
                int idx = cms3.pfjets_pfcandIndicies()[iJet][ipf];
                pf_pt.push_back(cms3.pfcands_p4()[idx].pt());
                pf_eta.push_back(cms3.pfcands_p4()[idx].eta());
                pf_phi.push_back(cms3.pfcands_p4()[idx].phi());
                pf_dr.push_back(ROOT::Math::VectorUtil::DeltaR(lep_p4, cms3.pfcands_p4()[idx]));
                pf_type.push_back(get_pf_type(idx));
                pf_id.push_back(cms3.pfcands_particleId()[idx]);
            }
        }
    }
    return make_tuple(pf_pt, pf_eta, pf_phi, pf_dr, pf_type, pf_id);
}

#endif
