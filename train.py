import uproot
import torch
import numpy as np
import glob

b_fatjet_inputs = np.empty([0,5])
b_weights = np.empty([0,1])
s_weights = np.empty([0,1])

for f in glob.glob("/ceph/mgais/svj/final/QCD*.root"):
    b_tree = uproot.open(f)['Events']
    b_filter_mask = np.array((b_tree["n_fatjet"].array()[:] > 0))
    b_pt = b_tree["FatJet_pt"].array()[b_filter_mask]
    b_pt_mask = (b_pt[:,0]>150)
    
    b_fatjet_inputs=np.append(b_fatjet_inputs, np.concatenate(
        (np.expand_dims(np.array(b_tree["FatJet_pt"].array()[b_filter_mask][b_pt_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_eta"].array()[b_filter_mask][b_pt_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_tau21"].array()[b_filter_mask][b_pt_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_tau32"].array()[b_filter_mask][b_pt_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_msoftdrop"].array()[b_filter_mask][b_pt_mask][:,0]),axis=1)),
        axis=-1
    ),axis=0)
    b_weights = np.append(b_weights, np.expand_dims(np.array(b_tree["evtweight"].array()[b_filter_mask][b_pt_mask]),axis=1),axis=0)
    print(f)
    print(np.isnan(b_fatjet_inputs).any())
    print(np.count_nonzero(np.isnan(b_fatjet_inputs)))
    print(len(np.array(b_tree["FatJet_tau2"].array()[b_filter_mask][b_pt_mask][:,0])) - np.count_nonzero(np.array(b_tree["FatJet_tau2"].array()[b_filter_mask][b_pt_mask][:,0])))
    print(len(np.array(b_tree["FatJet_tau1"].array()[b_filter_mask][b_pt_mask][:,0])) - np.count_nonzero(np.array(b_tree["FatJet_tau1"].array()[b_filter_mask][b_pt_mask][:,0])))
    

s_tree = uproot.open("/ceph/mgais/svj/final/signal.root")['Events']
print(s_tree.keys())

s_filter_mask = np.array((s_tree["n_fatjet"].array()[:] > 0))# & (s_tree["FatJet_pt"].array()[:,0] > 150))
s_pt = s_tree["FatJet_pt"].array()[s_filter_mask]
s_pt_mask = (s_pt[:,0]>150)
s_fatjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_eta"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau21"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau32"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_msoftdrop"].array()[s_filter_mask][s_pt_mask][:,0]),axis=1)),
    axis=-1
)
s_weights = np.append(s_weights, np.expand_dims(np.array(s_tree["evtweight"].array()[s_filter_mask][s_pt_mask]),axis=1),axis=0)


print("signal:")
print(np.isnan(s_fatjet_inputs).any())
print(np.count_nonzero(np.isnan(s_fatjet_inputs)))
print(len(np.array(s_tree["FatJet_tau2"].array()[s_filter_mask][s_pt_mask][:,0])) - np.count_nonzero(np.array(s_tree["FatJet_tau2"].array()[s_filter_mask][s_pt_mask][:,0])))
print(len(np.array(s_tree["FatJet_tau1"].array()[s_filter_mask][s_pt_mask][:,0])) - np.count_nonzero(np.array(s_tree["FatJet_tau1"].array()[s_filter_mask][s_pt_mask][:,0])))

print(s_fatjet_inputs)
print(b_fatjet_inputs)
print(b_weights)
print(s_weights)

print(s_filter_mask.shape)
print(type(s_fatjet_inputs))
print(s_fatjet_inputs.shape)

print(b_filter_mask.shape)
print(type(b_fatjet_inputs))
print(b_fatjet_inputs.shape)

print(b_weights.shape)

y_s = np.ones_like(s_fatjet_inputs[:,0])   #in example with [:,0] but why?
y_b = np.zeros_like(b_fatjet_inputs[:,0])

X = np.concatenate((s_fatjet_inputs, b_fatjet_inputs))           #combine signal and background
y = np.concatenate((y_s, y_b))
weights = np.concatenate((s_weights, b_weights))
idx = [i for i in range(len(X))]
import random
random.shuffle(idx)
X = X[idx]                                                       #randomize order
y = y[idx]
weights = weights[idx]
print(X.shape)
print(weights.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def train_lct(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=33)
    #clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    #y_pred = clf.predict_proba(X_test)
    #return y_pred[:,1], y_test
    
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    return y_pred[:,1], y_test

print("----------------")
print(np.isnan(X).any())
print(np.isnan(y).any())

y_pred, y_true = train_lct(X,y)

final_y_s = y_pred[y_true==1]
final_y_b = y_pred[y_true==0]

fig,ax = plt.subplots(figsize=(7,6))
bins=np.linspace(0,1,25)
plt.hist(final_y_s,histtype='step',density=True,label='Signal',bins=bins)
plt.hist(final_y_b,histtype='step',density=True,label='Background',bins=bins)
plt.legend()
fig.savefig('test.png')


