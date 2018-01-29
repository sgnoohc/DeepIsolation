//
#ifndef JETSUBSTRUCTUREVARIABLES_H
#define JETSUBSTRUCTUREVARIABLES_H

#include "fastjet/ClusterSequence.hh"
using namespace fastjet;

//#############################################################################
// Aggregate the dij distance parameters
void recursive_get_dijs(ClusterSequence cs, int jetidx, int depth, PseudoJet jet, vector<float>& dijs)
{
    if (depth == 0)
    {
        if (sorted_by_pt(cs.inclusive_jets()).size() == 0)
            return;
        PseudoJet j = sorted_by_pt(cs.inclusive_jets())[jetidx];
        dijs.clear();
        recursive_get_dijs(cs, jetidx, depth + 2, j, dijs);
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
            for (auto& p : jet.pieces())
            {
                recursive_get_dijs(cs, jetidx, depth + 2, p, dijs);
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
vector<float> compute_dijs(vector<LorentzVector> seeds, JetAlgorithm algo=kt_algorithm)
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
    recursive_get_dijs(cs, 0, 0, PseudoJet(), dijs);

    return dijs;
}

//##############################################################################
vector<float> get_dijs(LorentzVector lep_p4, JetAlgorithm algo=kt_algorithm, bool removecore=false)
{
    bool matched = false;
    for (unsigned int iJet = 0; iJet < cms3.pfjets_p4().size(); iJet++)
    {
        LorentzVector jet_p4 = cms3.pfjets_p4()[iJet];
        if (ROOT::Math::VectorUtil::DeltaR(jet_p4, lep_p4) < 0.4)
        {
            vector<LorentzVector> seeds;
            for (unsigned int ipf = 0; ipf < cms3.pfjets_pfcandIndicies()[iJet].size(); ++ipf)
            {
                int idx = cms3.pfjets_pfcandIndicies()[iJet][ipf];
                if (removecore)
                {
                    if (abs(lep_p4.pt() - cms3.pfcands_p4()[idx].pt()) > 0.001)
                        seeds.push_back(cms3.pfcands_p4()[idx]);
                }
                else
                {
                    seeds.push_back(cms3.pfcands_p4()[idx]);
                }
            }
            vector<float> rtn = compute_dijs(seeds, algo);
            return rtn;
        }
    }
    return vector<float>();
}

#endif
