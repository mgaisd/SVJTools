import uproot
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.seterr(all="ignore")
import glob
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


import os
import subprocess
import pandas as pd
from scipy.spatial import distance_matrix



#read in data

#s_tree = uproot.open("/ceph/mgais/svj/signal_genjetsv2/signal.root")['Events']
s_tree = uproot.open("/ceph/mgais/svj/test_ak4_vertex/signal.root")['Events']
#print(s_tree.keys())

s_filter_mask = np.array((s_tree["n_fatjet"].array()[:] > 0))
s_pt = s_tree["FatJet_pt"].array()[s_filter_mask]
s_pt_mask = (s_pt[:,0]>150)
s_nconst_mask = (s_tree["FatJet_nconst"].array()[s_filter_mask][s_pt_mask][:,0] > 2)
s_match_mask = (s_tree["FatJet_matched"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0] == 1)
s_nomatch_mask = (s_tree["FatJet_matched"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0] == 0)

#offline reconstruction
r_filter_mask = np.array((s_tree["n_recojet"].array()[:] > 0))
r_pt = s_tree["RecoJet_pt"].array()[r_filter_mask]
r_pt_mask = (r_pt[:,0]>150)
#r_nconst_mask = (s_tree["RecoJet_nconst"].array()[r_filter_mask][r_pt_mask][:,0] > 2) nconst not implemented

#imported genjets
g_filter_mask = np.array((s_tree["n_genjet"].array()[:] > 0))
g_pt = s_tree["GenJet_pt"].array()[g_filter_mask]
g_pt_mask = (g_pt[:,0]>150)
#r_nconst_mask = (s_tree["RecoJet_nconst"].array()[r_filter_mask][r_pt_mask][:,0] > 2) nconst not implemented

#manual genjets
gp_filter_mask = np.array((s_tree["n_genpjet"].array()[:] > 0))
gp_pt = s_tree["GenPJet_pt"].array()[gp_filter_mask]
gp_pt_mask = (gp_pt[:,0]>150)
#r_nconst_mask = (s_tree["RecoJet_nconst"].array()[r_filter_mask][r_pt_mask][:,0] > 2) nconst not implemented


#manual AK4
ak4_filter_mask = np.array((s_tree["n_AK4"].array()[:] > 0))
ak4_pt = s_tree["AK4_pt"].array()[ak4_filter_mask]
ak4_pt_mask = (ak4_pt[:,0]>150)
ak4_nconst_mask = (s_tree["AK4_nconst"].array()[ak4_filter_mask][ak4_pt_mask][:,0] > 2)
ak4_match_mask = (s_tree["AK4_matched"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][:,0] == 1)
ak4_nomatch_mask = (s_tree["AK4_matched"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][:,0] == 0)

#preclustered AK4
jet_filter_mask = np.array((s_tree["n_jet"].array()[:] > 0))
jet_pt = s_tree["Jet_pt"].array()[jet_filter_mask]
jet_pt_mask = (jet_pt[:,0]>150)
jet_nconst_mask = (s_tree["Jet_nConstituents"].array()[jet_filter_mask][jet_pt_mask][:,0] > 2)



s_fatjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_eta"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_phi"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1)),
    axis=-1
    )

s_genjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["GenJet_pt"].array()[g_filter_mask][g_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["GenJet_eta"].array()[g_filter_mask][g_pt_mask][:,0]),axis=1)),
    axis=-1
    )

s_genpjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["GenPJet_pt"].array()[gp_filter_mask][gp_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["GenPJet_eta"].array()[gp_filter_mask][gp_pt_mask][:,0]),axis=1)),
    axis=-1
    )

s_recojet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["RecoJet_pt"].array()[r_filter_mask][r_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["RecoJet_eta"].array()[r_filter_mask][r_pt_mask][:,0]),axis=1)),
    axis=-1
    )


#jetmatching_inputs = np.concatenate(
#matched = np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][s_match_mask][:,0]),axis=1)
#non_matched = np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][s_nomatch_mask][:,0]),axis=1)
#matched_gen = np.expand_dims(np.array(s_tree["GenJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][s_match_mask][:,0]),axis=1)
#non_matched_gen = np.expand_dims(np.array(s_tree["GenJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][s_nomatch_mask][:,0]),axis=1)

ak4 = np.expand_dims(np.array(s_tree["Jet_pt"].array()[jet_filter_mask][jet_pt_mask][jet_nconst_mask][:,0]),axis=1)
ak4_manual = np.expand_dims(np.array(s_tree["AK4_pt"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][:,0]),axis=1)
ak4_manual_match = np.expand_dims(np.array(s_tree["AK4_pt"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][ak4_match_mask][:,0]),axis=1)
ak4_manual_nomatch = np.expand_dims(np.array(s_tree["AK4_pt"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][ak4_nomatch_mask][:,0]),axis=1)

ak4_eta = np.expand_dims(np.array(s_tree["Jet_eta"].array()[jet_filter_mask][jet_pt_mask][jet_nconst_mask][:,0]),axis=1)
ak4_manual_eta = np.expand_dims(np.array(s_tree["AK4_eta"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][:,0]),axis=1)

ak4_nconst = np.expand_dims(np.array(s_tree["Jet_nConstituents"].array()[jet_filter_mask][jet_pt_mask][jet_nconst_mask][:,0]),axis=1)
ak4_manual_nconst = np.expand_dims(np.array(s_tree["AK4_nconst"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][:,0]),axis=1)
ak4_manual_nconst_match = np.expand_dims(np.array(s_tree["AK4_nconst"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][ak4_match_mask][:,0]),axis=1)
ak4_manual_nconst_nomatch = np.expand_dims(np.array(s_tree["AK4_nconst"].array()[ak4_filter_mask][ak4_pt_mask][ak4_nconst_mask][ak4_nomatch_mask][:,0]),axis=1)


#print(s_fatjet_inputs)
#print(s_fatjet_inputs[:,0])


#
# PLOTS
#

#plots for jetmatching
#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(100,1000,30)
#plt.hist(matched,histtype='step',density=False,label='matched Scouting AK8CHS',bins=bins)
#plt.hist(matched_gen,histtype='step',density=False,label='matched GenJets',bins=bins)
#plt.xlabel(r"leading jet $p_T$ in GeV")
#plt.ylabel("#")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/matched_scouting_vs_gen.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/matched_scouting_vs_gen.pdf',bbox_inches='tight')

#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(100,1000,30)
#plt.hist(non_matched,histtype='step',density=False,label='non-matched Scouting AK8CHS',bins=bins)
#plt.hist(non_matched_gen,histtype='step',density=False,label='non-matched GenJets',bins=bins)
#plt.xlabel(r"leading jet $p_T$ in GeV")
#plt.ylabel("#")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/non_matched_scouting_vs_gen.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/non_matched_scouting_vs_gen.pdf',bbox_inches='tight')

#plot for ak4 preclustered vs manual
#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(100,1000,30)
#plt.hist(ak4,histtype='step',density=False,label='pre-clustered AK4 Scouting Jets',bins=bins)
#plt.hist(ak4_manual_match,histtype='step',density=False,label='matched manually clustered AK4 Scouting Jets',bins=bins)
#plt.hist(ak4_manual_nomatch,histtype='step',density=False,label='non-matched manually clustered AK4 Scouting Jets',bins=bins)
#plt.xlabel(r"leading jet $p_T$ in GeV")
#plt.ylabel("Norm. to unit area")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_pt_matched_novertex.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_pt_matched_novertex.pdf',bbox_inches='tight')

#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(-2.5,2.5,30)
#plt.hist(ak4_eta,histtype='step',density=True,label='pre-clustered AK4 Scouting Jets',bins=bins)
#plt.hist(ak4_manual_eta,histtype='step',density=True,label='manually clustered AK4 Scouting Jets',bins=bins)
#plt.xlabel(r"leading jet $\eta$")
#plt.ylabel("Norm. to unit area")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_eta.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_eta.pdf',bbox_inches='tight')

#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(0,100,30)
#plt.hist(ak4_nconst,histtype='step',density=False,label='pre-clustered AK4 Scouting Jets',bins=bins)
#plt.hist(ak4_manual_nconst_match,histtype='step',density=False,label='matched manually clustered AK4 Scouting Jets',bins=bins)
#plt.hist(ak4_manual_nconst_nomatch,histtype='step',density=False,label='non-matched manually clustered AK4 Scouting Jets',bins=bins)
#plt.xlabel(r"leading jet number of constituents")
#plt.ylabel("Norm. to unit area")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_nconst_matched_novertex.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/ak4_comparison/ak4_nconst_matched_novertex.pdf',bbox_inches='tight')





#plot 1D histogram 
fig,ax = plt.subplots(figsize=(7,6))
bins=np.linspace(100,1000,30)
plt.hist(s_fatjet_inputs[:,0],histtype='step',density=True,label='Scouting AK8CHS',bins=bins)
plt.hist(s_genjet_inputs[:,0],histtype='step',density=True,label='GenJets',bins=bins)
plt.hist(s_genpjet_inputs[:,0],histtype='step',density=True,label='GenJets (manually clustered)',bins=bins)
plt.hist(s_recojet_inputs[:,0],histtype='step',density=True,label='Offline AK8CHS',bins=bins)
plt.xlabel(r"$p_T$ in GeV")
plt.ylabel("Norm. to unit area")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/scouting_offline_gen_comparison/pt_novertex.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/scouting_offline_gen_comparison/pt_novertex.pdf',bbox_inches='tight')

#plot eta distribution
#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(-4,4,30)
#plt.hist(s_fatjet_inputs[:,1],histtype='step',density=True,label='Scouting AK8CHS',bins=bins)
#plt.hist(s_genjet_inputs[:,1],histtype='step',density=True,label='GenJets',bins=bins)
#plt.hist(s_genpjet_inputs[:,1],histtype='step',density=True,label='GenJets (manually clustered)',bins=bins)
#plt.hist(s_recojet_inputs[:,1],histtype='step',density=True,label='Offline AK8CHS',bins=bins)
#plt.xlabel(r"$\eta$")
#plt.ylabel("Norm. to unit area")
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/scouting_offline_gen_comparison/eta.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/scouting_offline_gen_comparison/eta.pdf',bbox_inches='tight')


#plot 2D histogram
#fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(0,1000,30)
#plt.hist2d(s_fatjet_inputs[:,0], s_fatjet_inputs[:,1],density=True,bins=bins)
#plt.xlabel(r"FatJet $p_T$ in GeV")
#plt.ylabel(r"GenParticleJet $p_T$ in GeV")
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/2D_genparticle_vs_fatjet_pt.png',bbox_inches='tight',dpi=300)
#fig.savefig('/etpwww/web/mgais/public_html/svj/figs/2D_genparticle_vs_fatjet_pt.pdf',bbox_inches='tight')
