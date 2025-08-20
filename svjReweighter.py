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
from lund_utils.Utils import *
import math

def run_reweighting(year, constituents, jets, nDarkHadronsPerJet, nProngsPerJet, nevts_tot):
    # Lund systematic setup
    nToys = 100
    nSys = 10

    nevts_batch = nevts_tot

    LP_weights = np.zeros(nevts_batch)
    LP_mjj_check = np.zeros(nevts_batch)
    LP_weights_stat_var = np.zeros((nevts_batch, nToys))
    LP_weights_pt_var = np.zeros((nevts_batch, nToys))
    LP_weights_sys_var = np.zeros((nevts_batch, nSys))

    f_ratio_current = ROOT.TFile.Open("../LundReweighting/data/ratio_"+str(year)+".root")

    LP_rws = LundReweighter( f_ratio = f_ratio_current )

    #Noise used to generated smeared ratio's based on stat unc
    np.random.seed(123)
    rand_noise = np.random.normal(size = (nToys, LP_rws.h_ratio.GetNbinsX(), LP_rws.h_ratio.GetNbinsY(), LP_rws.h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size = (nToys, LP_rws.h_ratio.GetNbinsY(), LP_rws.h_ratio.GetNbinsZ(), 3))
    #print('rand', rand_noise[0,0,0,:5])

    out = LP_rws.get_all_weights(constituents, None, jets, do_sys_weights = True, distortion_sys = True, rand_noise = rand_noise, pt_rand_noise = pt_rand_noise, normalize = True, pf_cands_PtEtaPhiE_format = True, nDark = nDarkHadronsPerJet, nProngs = nProngsPerJet)

    return out

def calculate_lund_weights(events, mc_year):

    # number of events in current sample
    nevts_tot = ak.size(events["EvtNum"])
    # print(str(nevts_tot) + " events in sample")

    # mask to only svj categories
    mask = events["JetsAK8"].genIndex >= 0
    gen_jet_index = events["JetsAK8"].genIndex[mask]

    genJets = events["GenJetsAK8"][gen_jet_index]

    hv_category = genJets.hvCategory
    maskhv = (hv_category == 3) | (hv_category == 9) | (hv_category == 11) | (hv_category == 5) | (hv_category == 7) | (hv_category == 13)

    ak8genjets = genJets[maskhv]
    ak8jets = events["JetsAK8"]
    ak8jets_svj = ak8jets[maskhv]

    # print(str(len(ak.flatten(ak.nan_to_num(ak8jets.pt, nan=0)))) + " matched signal jets in sample.")

    maskMatched = ak8jets_svj.darkHadronJets.constituentsAssignedSecond != 1

    # print(ak.sum(ak8jets_svj.darkHadronJets.constituentsAssignedSecond), " number of constituents not used (assigned to second closest dark hadron.")

    ak8jets_constituents_pdgid = ak8jets_svj.darkHadronJets.constituentsPdgid[maskMatched]
    ak8jets_constituents = ak8jets_svj.darkHadronJets.constituents[maskMatched]

    constituents_pt = ak.nan_to_num(ak8jets_constituents.pt,nan=0)
    constituents_eta = ak.nan_to_num(ak8jets_constituents.eta,nan=0)
    constituents_phi = ak.nan_to_num(ak8jets_constituents.phi,nan=0)
    constituents_E = ak.nan_to_num(ak8jets_constituents.E,nan=0)

    jets_pt = ak.nan_to_num(ak8jets_svj.pt ,nan=0)
    jets_eta = ak.nan_to_num(ak8jets_svj.eta ,nan=0)
    jets_phi = ak.nan_to_num(ak8jets_svj.phi ,nan=0)
    jets_E = ak.nan_to_num(ak8jets_svj.E ,nan=0)

    #flatten to get a 1D list of the darkHadronJet constituents and the jets- will have to deal with postprocessing this later
    jetsFlat = ak.zip([ak.flatten(jets_pt), ak.flatten(jets_eta), ak.flatten(jets_phi), ak.flatten(jets_E) ] )
    constituentsFlat = ak.zip( [ak.flatten(ak.flatten(constituents_pt)), ak.flatten(ak.flatten(constituents_eta)), ak.flatten(ak.flatten(constituents_phi)), ak.flatten(ak.flatten(constituents_E)) ] )

    nDarkHadronsPerJet = ak.num(constituents_pt,axis=2)
    nProngsPerJet = nDarkHadronsPerJet*2
    nJetsPerEvent = ak.num(jets_pt, axis=1)
    nProngsPerEvent = ak.flatten(nProngsPerJet)

    # returns a weight per jet
    output = run_reweighting(mc_year, constituentsFlat, jetsFlat, nDarkHadronsPerJet, nProngsPerJet, nevts_tot)

    for k in output.keys():
        if k in ['subjet_pts_perDarkHadron', 'subjet_weights', 'lp_idxs', 'splitting_weights', 'nSplittings', 'n_prongs', 'subjet_pts', 'bad_match', 'reclust_still_bad_match', 'reclust_nom', 'reclust_prongs_up', 'reclust_prongs_down', 'nom_noNorm']: continue
        elif k in ["stat_vars", "pt_vars"]:
            # Unflatten to (nEvents, nJetsPerEvent, nToys)
            stat_vars = ak.unflatten(output[k], nJetsPerEvent)
            # Take mean and std over the toys axis (axis=2)
            weights_up = ak.mean(stat_vars, axis=2) + ak.std(stat_vars, axis=2)
            weights_down = ak.mean(stat_vars, axis=2) - ak.std(stat_vars, axis=2)
            event_weights_up = ak.prod(weights_up, axis=-1)
            event_weights_down = ak.prod(weights_down, axis=-1)
            k = k.capitalize()
            events[f"lundWeight{k.replace('_vars', '')}Up"] = np.clip(event_weights_up, 0,5)
            events[f"lundWeight{k.replace('_vars', '')}Down"] = np.clip(event_weights_down, 0,5)
        else:
            print(k)
            weights = ak.unflatten(output[k], nJetsPerEvent)
            event_weights = ak.prod(weights, axis=-1)
            k = k.capitalize().replace('up', 'Up').replace('down', 'Down').replace('distortion', 'Distortion')
            events["lundWeight"+k.replace('_','')] = np.clip(event_weights, 0,5)

    return events, maskempty
