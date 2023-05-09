import uproot
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os
import subprocess
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import h5py
import awkward as ak
import pickle
import pandas as pd
from scipy.spatial import distance_matrix



b_fatjet_inputs = np.empty([0,5])
b_weights = np.empty([0,1])
s_weights = np.empty([0,1])

for f in glob.glob("/ceph/mgais/svj/finalv2/QCD*.root"):
    b_tree = uproot.open(f)['Events']
    b_filter_mask = np.array((b_tree["n_fatjet"].array()[:] > 0))
    b_pt = b_tree["FatJet_pt"].array()[b_filter_mask]
    b_pt_mask = (b_pt[:,0]>150)
    b_nconst_mask = (b_tree["FatJet_nconst"].array()[b_filter_mask][b_pt_mask][:,0] > 2)    
    b_fatjet_inputs=np.append(b_fatjet_inputs, np.concatenate(
        (np.expand_dims(np.array(b_tree["FatJet_pt"].array()[b_filter_mask][b_pt_mask][b_nconst_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_eta"].array()[b_filter_mask][b_pt_mask][b_nconst_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_tau21"].array()[b_filter_mask][b_pt_mask][b_nconst_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_tau32"].array()[b_filter_mask][b_pt_mask][b_nconst_mask][:,0]),axis=1),
         np.expand_dims(np.array(b_tree["FatJet_msoftdrop"].array()[b_filter_mask][b_pt_mask][b_nconst_mask][:,0]),axis=1)),
        axis=-1
    ),axis=0)
    b_weights = np.append(b_weights, np.expand_dims(np.array(b_tree["evtweight"].array()[b_filter_mask][b_pt_mask][b_nconst_mask]),axis=1),axis=0)
    print(f)
   # print(np.isnan(b_fatjet_inputs).any())
   # print(np.count_nonzero(np.isnan(b_fatjet_inputs)))
    

s_tree = uproot.open("/ceph/mgais/svj/final/signal.root")['Events']
#print(s_tree.keys())

s_filter_mask = np.array((s_tree["n_fatjet"].array()[:] > 0))# & (s_tree["FatJet_pt"].array()[:,0] > 150))
s_pt = s_tree["FatJet_pt"].array()[s_filter_mask]
s_pt_mask = (s_pt[:,0]>150)
s_nconst_mask = (s_tree["FatJet_nconst"].array()[s_filter_mask][s_pt_mask][:,0] > 2)
s_fatjet_inputs =  np.concatenate(
    (np.expand_dims(np.array(s_tree["FatJet_pt"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_eta"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau21"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_tau32"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1),
     np.expand_dims(np.array(s_tree["FatJet_msoftdrop"].array()[s_filter_mask][s_pt_mask][s_nconst_mask][:,0]),axis=1)),
    axis=-1
)
s_weights = np.append(s_weights, np.expand_dims(np.array(s_tree["evtweight"].array()[s_filter_mask][s_pt_mask][s_nconst_mask]),axis=1),axis=0)


#print("signal:")
#print(np.isnan(s_fatjet_inputs).any())
#print(np.count_nonzero(np.isnan(s_fatjet_inputs)))

#print(s_fatjet_inputs)
#print(b_fatjet_inputs)
#print(b_weights)
#print(s_weights)

#print(s_filter_mask.shape)
#print(type(s_fatjet_inputs))
#print(s_fatjet_inputs.shape)

#print(b_filter_mask.shape)
#print(type(b_fatjet_inputs))
#print(b_fatjet_inputs.shape)

#print(b_weights.shape)

sum_b_weights=np.sum(b_weights)
n_s=np.sum(s_weights)

s_weights[s_weights == 1] = sum_b_weights/n_s
#print(np.sum(b_weights))
#print(np.sum(s_weights))

y_s = np.ones_like(s_fatjet_inputs[:,0])   
y_b = np.zeros_like(b_fatjet_inputs[:,0])

X = np.concatenate((s_fatjet_inputs, b_fatjet_inputs))           #combine signal and background
y = np.concatenate((y_s, y_b))
weights = np.concatenate((s_weights, b_weights))
idx = [i for i in range(len(X))]

random.shuffle(idx)
X = X[idx]                                                       #randomize order
y = y[idx]
weights = weights[idx]
print(X.shape)
print(weights.shape)
print(y.shape)



#print("NaN values in final data and truth set:")
#print(np.isnan(X).any())
#print(np.isnan(y).any())

#Split up data into training and test data sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.30, random_state=33)
X_train_np, X_test_np = np.split(X, [int(len(X)*0.7)])
y_train_np, y_test_np = np.split(y, [int(len(y)*0.7)])
weights_train_np, weights_test_np = np.split(weights, [int(len(weights)*0.7)])



#torch expects tensor as input
X_train = torch.from_numpy(X_train_np).float()
X_test = torch.from_numpy(X_test_np).float()
y_train = torch.from_numpy(y_train_np).float()
y_test = torch.from_numpy(y_test_np).float()
weights_train = torch.from_numpy(weights_train_np).float()
weights_test = torch.from_numpy(weights_test_np).float()

#--------------------------------------------------------------------------------
#training
device = "cuda" if torch.cuda.is_available() else "cpu"
devide = 'cpu'
print(f"Using {device} device")


model = nn.Sequential(nn.Linear(X.shape[1], 25),
                      nn.ReLU(),
                      nn.Linear(25, 5),
                      nn.ReLU(),
                      nn.Linear(5, 1),
                      nn.Sigmoid())

print("XXXX")
print(weights_train.shape)
print(torch.flatten(weights_train).shape)
def weighted_loss(y,y_hat,w):
    #loss_function = nn.BCEWithLogitsLoss(weight=torch.flatten(weights_train).unsqueeze(1))
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    return (loss_function(y, y_hat)*w).mean()

#loss_function = nn.BCELoss()
#loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    pred_y = model(X_train)
    #print(pred_y)

    #print("XXXXXX")
    #print(pred_y.shape)
    #print(torch.flatten(pred_y).shape)
    #print(y_train.shape)

    
    #loss = loss_function(torch.flatten(pred_y), y_train)

    print(torch.flatten(pred_y).shape)
    print(weights_train.shape)
    print(y_train.shape)
    
    loss = weighted_loss(torch.flatten(pred_y), y_train, torch.flatten(weights_train))


    #losses.append(loss.item())
    
    #print(loss.item())                                                                                                                                                                                   

    model.zero_grad()
    loss.backward()

    optimizer.step()

    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    with torch.no_grad():
        pred_y = model(X_test)
        #loss = loss_function(torch.flatten(pred_y), y_test)
        loss = weighted_loss(torch.flatten(pred_y), y_test, torch.flatten(weights_test))
        return loss.item(), torch.flatten(pred_y), y_test

@torch.no_grad()
def val(X_test):
    model.eval()
    with torch.no_grad():
        pred_y = model(X_test)
        return pred_y


def early_stopping(train_loss, validation_loss, min_delta, tolerance):

    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
          return True

#modify from here
                      
best_val_loss = 1e9

loss_dict = {'train':[],'val':[]}

last_ten = []

for epoch in range(3000):
    train_loss = train()

    val_loss, pred_y, y_test = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
    epoch, train_loss, val_loss))

    loss_dict['train'].append(train_loss)
    loss_dict['val'].append(val_loss)


    #print(pred_y)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        state_dicts = {'model':model.state_dict(),
                       'opt':optimizer.state_dict()}

        torch.save(state_dicts, 'best-epoch.pt')

    #if early_stopping(loss_dict['train'], loss_dict['val'], min_delta=10, tolerance = 20):                                                                                                               
    #  print("We are at epoch:", epoch)                                                                                                                                                                   
    #  break                                                                                                                                                                                              


    if len(last_ten) > 9:
        last_ten.pop(0)
        last_ten.append(val_loss)

        criterion = (np.max(last_ten) - np.min(last_ten))/np.max(last_ten)
        #print(np.max(last_ten))                                                                                                                                                                          
        #print(np.min(last_ten))                                                                                                                                                                          
        #print(criterion)                                                                                                                                                                                 
        if criterion < 0.00003:
            break
    else:
        last_ten.append(val_loss)


df = pd.DataFrame.from_dict(loss_dict)
print(df)
#df.to_csv("losses_t0p1_dim512.csv")



final_y_s = pred_y[y_test==1].numpy()
final_y_b = pred_y[y_test==0].numpy()
final_w_s = weights_test[y_test==1].flatten()
final_w_b = weights_test[y_test==0].flatten()



#plot 
fig,ax = plt.subplots(figsize=(7,6))
#bins=np.linspace(0,1,25)
plt.hist(final_y_s,weights=final_w_s,histtype='step',density=True,label='Signal')#,bins=bins)
plt.hist(final_y_b,weights=final_w_b,histtype='step',density=True,label='Background')#,bins=bins)
plt.legend()
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/test.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/test.pdf',bbox_inches='tight')


