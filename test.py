import uproot
import torch
import numpy as np
import glob


s_tree = uproot.open("/ceph/mgais/svj/final/signal.root")['Events']
print(s_tree.keys())

s_filter_mask = np.array((s_tree["n_fatjet"].array()[:] > 0))# & (s_tree["FatJet_pt"].array()[:,0] > 150))                                                                                                 
s_pt = s_tree["FatJet_pt"].array()[s_filter_mask]
#s_photon_pt = s_tree["Photon_pt"].array()[s_filter_mask]
s_pt_mask = (s_pt[:,0]>150)# & (s_photon_pt[:,0]>50)
s_fatjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_eta"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau21"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau32"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_msoftdrop"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
    # np.expand_dims(np.array(s_tree["Photon_pt"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_nconst"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1)),
    axis=-1
)

print(np.isnan(s_fatjet_inputs).any())
print(len(np.array(s_tree["FatJet_tau2"].array()[s_filter_mask][s_pt_mask][:,0])) - np.count_nonzero(np.array(s_tree["FatJet_tau2"].array()[s_filter_mask][s_pt_mask][:,0])))
print(len(np.array(s_tree["FatJet_tau1"].array()[s_filter_mask][s_pt_mask][:,0])) - np.count_nonzero(np.array(s_tree["FatJet_tau1"].array()[s_filter_mask][s_pt_mask][:,0])))
print(np.argwhere(np.isnan(s_fatjet_inputs)))
print(s_fatjet_inputs[np.argwhere(np.isnan(s_fatjet_inputs))[:,0]])
print(s_tree["Photon_pt"].array()[np.argwhere(np.isnan(s_fatjet_inputs))[:,0]])
