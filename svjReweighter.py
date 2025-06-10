import awkward as ak
import numpy as np
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory #, TreeMakerSchema
import treemaker
from treemaker import TreeMakerSchema
import fastjet

import sys, os
sys.path.insert(0, '')
sys.path.append("../")
from utils.Utils import *
import math


def fill_hist(var,label,samples):
    hists = {}
    for sample in samples:
        h = (
            hist.Hist.new
            .Reg(50, 0, 1000, label=label)
            .Double()
        )
        h.fill(var) #,weight=0.5)
        hists[sample["legname"] + ' ' + label] = h
        #stat = hist.accumulators.Mean().fill(var)
        print("{} {}".format("jetpt", sample["rinv"]))
    return hists

def fill_hist_w(var,label,samples):
    hists = {}
    for sample in samples:
        h = (
            hist.Hist.new
            .Reg(50, 0, 5, label=label)
            .Double()
        )
        h.fill(var)
        hists[sample["legname"] + ' ' + label] = h
        #stat = hist.accumulators.Mean().fill(var)
        print("{} {}".format("weights", sample["rinv"]))
    return hists


samples = [ {"rinv": 3, "legname": r"$m_{\mathrm{dark}} = 20$ GeV, $r_{\mathrm{inv}} = 0.3$", "fname":"/uscms/home/easmith/svj/CMSSW_10_6_29_patch1/src/TreeMaker/run/current-pr/SVJ_UL2018_t-channel_mMed-2000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_13TeV-madgraphMLM-pythia8_n-1000_RA2AnalysisTree.root"}
]

for sample in samples:

    sample["events"] = NanoEventsFactory.from_root(
        file=sample["fname"],
        treepath="TreeMaker2/PreSelection",
        schemaclass=TreeMakerSchema,
        metadata={"dataset": sample["legname"]},
    ).events()
    
    events = sample["events"]
   
    # number of events in current sample
    nevts_tot = ak.size(sample["events"]["EvtNum"])
    print(str(nevts_tot) + " events in sample")
    
    
    events = sample["events"]
    
    # # #limit to 1000 events to start
    # if nevts_tot > 1000: 
    #    events = sample["events"][0:1000]
    #    nevts_tot = 1000
    #    print("Using only first 1000 events!")


    # mask to only svj categories
    mask = events["JetsAK8"].genIndex >= 0
    gen_jet_index = events["JetsAK8"].genIndex[mask]

    # print(gen_jet_index)
    # print(ak.count(events["GenJetsAK8"].pt, axis=1))
        
    genJets = events["GenJetsAK8"][gen_jet_index] 
   
    hv_category = genJets.hvCategory
    #maskGenMatched = events["JetsAK8"].genIndex >= 0
    maskhv = (hv_category == 3) | (hv_category == 9) | (hv_category == 11) | (hv_category == 5) | (hv_category == 7) | (hv_category == 13)
    
    # print("len hv_category", len(hv_category))
    # print("len hv category mask", len(maskhv))

    ak8genjets = genJets[maskhv]
    
    ak8genjets.darkHadronJets.constituents 
    
    ak8jets = events["JetsAK8"]
    ak8jets_svj = ak8jets[maskhv]

    print(str(len(ak.flatten(ak.nan_to_num(ak8jets.pt, nan=0)))) + " matched signal jets in sample.")
   
    maskMatched = ak8jets_svj.darkHadronJets.constituentsAssignedSecond != 1

    print("constituens\n", ak.count(ak8jets_svj.darkHadronJets.constituents.pt, axis=3)[127:131])
    print("assigned second\n", ak.count(ak8jets_svj.darkHadronJets.constituentsAssignedSecond,axis=3)[127:131])
    print("pdgid\n", ak.count(ak8jets_svj.darkHadronJets.constituentsPdgid,axis=3)[127:131])
    print()
    print("constituens\n", ak8jets_svj.darkHadronJets.constituents[127][0])
    print("assigned second\n", ak8jets_svj.darkHadronJets.constituentsAssignedSecond[127][0])
    print("pdgid\n", ak8jets_svj.darkHadronJets.constituentsPdgid[127][0])

    # count = 0
    # for x in ak8jets_svj.darkHadronJets.constituentsAssignedSecond:
    #     cc = 0
    #     for xx in x:           
    #         y = ak.count(ak8jets_svj.darkHadronJets.constituents[count][cc])
    #         xxx = ak.count(xx)
    #         if (xxx != y):
    #             print("List of constituent=s and list of if assigned second have different sizes. constituents: " + str(y) + " assignedsecond: " + str(xxx))
    #         cc+=1
    #     count+=1 

    print(ak.sum(ak8jets_svj.darkHadronJets.constituentsAssignedSecond), " number of constituents not used (assigned to second closest dark hadron.")

    ak8jets_final_pdgid = ak8jets_svj.darkHadronJets.constituentsPdgid[maskMatched]         
    ak8jets_final = ak8jets_svj.darkHadronJets.constituents[maskMatched]         
         
    constituents_pt = ak.nan_to_num(ak8jets_final.pt,nan=0)
    constituents_eta = ak.nan_to_num(ak8jets_final.eta,nan=0)
    constituents_phi = ak.nan_to_num(ak8jets_final.phi,nan=0)
    constituents_E = ak.nan_to_num(ak8jets_final.E,nan=0)

    jets_pt = ak.nan_to_num(ak8jets_svj.pt ,nan=0)
    jets_eta = ak.nan_to_num(ak8jets_svj.eta ,nan=0)
    jets_phi = ak.nan_to_num(ak8jets_svj.phi ,nan=0)
    jets_E = ak.nan_to_num(ak8jets_svj.E ,nan=0)
    
    print("flattened jet pt", ak.flatten(jets_pt))
    #flatten to get a 1D list of the darkHadronJet constituents and the jets- will have to deal with postprocessing this later 
    jetsFlat = ak.zip([ak.flatten(jets_pt), ak.flatten(jets_eta), ak.flatten(jets_phi), ak.flatten(jets_E) ] )
    constituentsFlat = ak.zip( [ak.flatten(ak.flatten(constituents_pt)), ak.flatten(ak.flatten(constituents_eta)), ak.flatten(ak.flatten(constituents_phi)), ak.flatten(ak.flatten(constituents_E)) ] )
    print(constituentsFlat)

    print(str(len(jetsFlat)) + " jets in samples after hv mask")
    #prongs per jet = nDarkHadronJetsPerJet * 2
    # outer list of jets ak.flatten(events["constituents_pt"])
    jetConstituents = constituents_pt
    nDarkHadronsPerJet = ak.num(jetConstituents,axis=1)
    nProngsPerJet = nDarkHadronsPerJet*2
    
    # Lund systematic setup
    nToys = 100
    nSys = 10
   
    nevts_batch = nevts_tot
   
    LP_weights = np.zeros(nevts_batch)
    LP_mjj_check = np.zeros(nevts_batch)
    LP_weights_stat_var = np.zeros((nevts_batch, nToys))
    LP_weights_pt_var = np.zeros((nevts_batch, nToys))
    LP_weights_sys_var = np.zeros((nevts_batch, nSys))
    
    f_ratio18 = ROOT.TFile.Open("data/ratio_2018.root")
    f_ratio17 = ROOT.TFile.Open("data/ratio_2017.root")
    f_ratio16 = ROOT.TFile.Open("data/ratio_2016.root")
      
    LP_rws = [ LundReweighter(f_ratio = f_ratio16 ), LundReweighter(f_ratio = f_ratio17 ), LundReweighter(f_ratio = f_ratio18 )]

    #Noise used to generated smeared ratio's based on stat unc
    np.random.seed(123)
    rand_noise = np.random.normal(size = (nToys, LP_rws[0].h_ratio.GetNbinsX(), LP_rws[0].h_ratio.GetNbinsY(), LP_rws[0].h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size = (nToys, LP_rws[0].h_ratio.GetNbinsY(), LP_rws[0].h_ratio.GetNbinsZ(), 3))
    print('rand', rand_noise[0,0,0,:5])

    #def get_all_weights(self, pf_cands, gen_parts_eta_phi, ak8_jets, gen_parts_pdg_ids = None, do_sys_weights = True, distortion_sys = True, nToys = 100, rand_noise = None, pt_rand_noise = None, normalize = True, pt_norm = True, pf_cands_PtEtaPhiE_format = False):
    # gen_parts_eta_phi = -1, no number of generator level quarks since we already have subjets
   
   
    out = LP_rws[2].get_all_weights(constituentsFlat, None, jetsFlat, do_sys_weights = True, distortion_sys = True, rand_noise = rand_noise, pt_rand_noise = pt_rand_noise, normalize = True, pf_cands_PtEtaPhiE_format = True, nDark = nDarkHadronsPerJet, nProngs = nProngsPerJet)
    
    flatJetPt = ak.to_numpy(ak.flatten(jets_pt))

    weightedFlatJets = [j*w for j, w in zip(flatJetPt,out['nom'])]
    weightedFlatJetsNoNorm = [j*w for j, w in zip(flatJetPt,out['nom_noNorm'])]
   
    # maskLow = flatJetPt < 170
    # maskHigh = flatJetPt > 170
    print("Number of weights: ", len(out['nom']))
    histnames_w = [
        (out['nom'],r"Lund weights"),
        # (out['nom'][maskLow],r"Lund weights for Jet $p_T$ < 170 [GeV]"),
        # (out['nom'][maskHigh],r"Lund weights for Jet $p_T$ > 170 [GeV]"),
    ]
    
    histnames = [
        (flatJetPt,r"Jet $p_T$ [GeV]"),
        (weightedFlatJets,r"reweighted Jet $p_T$ [GeV]"),
        (weightedFlatJetsNoNorm,r"reweighted(no normalization) Jet $p_T$ [GeV]"),
    ]
    allhists = []
    for v,l in histnames:
        allhists.append((v,fill_hist(v,l,samples)))

    allhists_w = []
    for v,l in histnames_w:
        allhists_w.append((v,fill_hist_w(v,l,samples)))
    
    #plt.style.use(hep.style.CMS)
    mpl.rcParams.update({
        "axes.labelsize" : 18,
        "legend.fontsize" : 16,
        "xtick.labelsize" : 14,
        "ytick.labelsize" : 14,
        "font.size" : 18,
        "legend.frameon": True,
    })
    # based on https://github.com/mpetroff/accessible-color-cycles
    # red, blue, mauve, orange, purple, gray, 
    #colors = ["#e42536", "#5790fc", "#964a8b", "#f89c20", "#7a21dd", "#9c9ca1"]
    colors = ["#e42536", "#7a21dd", "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    styles = ['--','-.',':',]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
    
    fig, ax = plt.subplots(figsize=(8,6))
    for hname,hists in allhists:
        for l,h in hists.items():
            hep.histplot(h,density=True,ax=ax,label=l)
        #ax.set_xlim(0,600)
        #ax.set_yscale("log")
        ax.set_ylabel("Arbitrary units")
        #ax.set_ylim(1,2e5)
    ax.legend(framealpha=0.5, prop={'size': 10})
    #plt.xlabel("N-subjettiness")
    plt.savefig('{}.pdf'.format("jetpt"),bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8,6))
    for hname,hists in allhists_w:
        for l,h in hists.items():
            hep.histplot(h,density=False,ax=ax,label=l)
        #ax.set_xlim(0,600)
        #ax.set_yscale("log")
        ax.set_ylabel("Arbitrary units")
        #ax.set_ylim(1,2e5)
        ax.legend(framealpha=0.5, prop={'size': 10})
    #plt.xlabel("N-subjettiness")
    plt.savefig('{}.pdf'.format("weights"),bbox_inches='tight')

# fig, ax = plt.subplots(figsize=(8,6))
# for hname,hists in allhists[3:5]:
#     for l,h in hists.items():
#         hep.histplot(h,density=True,ax=ax,label=l)
#     ax.set_xlim(0,1)
#     #ax.set_yscale("log")
#     ax.set_ylabel("Arbitrary units")
#     #ax.set_ylim(1,2e5)
#     ax.legend(framealpha=0.5, prop={'size': 10})
# plt.xlabel("N-subjettiness Ratios")
# plt.savefig('{}.pdf'.format(hname),bbox_inches='tight')
