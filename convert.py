import numpy as np
import h5py
from sys import argv
from re import sub
import uproot as uproot
import sys
import tqdm
import random

def padarray(A, size):
    t = size - len(A)

    if len(A) > size:
        return A[:size]
    else:
        return np.pad(A, pad_width=(0, t), mode='constant')

infile = sys.argv[1]
issignal = bool(int(sys.argv[2]))
try:
    maxevents = int(sys.argv[3]) ### also means random shuffling
except:
    maxevents = -1

branches = ['FatJet1Const_pt','FatJet1Const_eta','FatJet1Const_phi','FatJet1Const_mass','FatJet1Const_charge','FatJet1Const_pdgID','FatJet2Const_pt','FatJet2Const_eta','FatJet2Const_phi','FatJet2Const_mass','FatJet2Const_charge','FatJet2Const_pdgID','evtweight']

ifile = uproot.open(infile)['events']
n_fatjet = ifile['n_fatjet'].array() > 1
ht = ifile['ht'].array()[n_fatjet] > 500
MT = ifile['MT'].array()[n_fatjet] > 720
DeltaEta = abs(ifile['FatJet_eta'].array()[n_fatjet][:,0] - ifile['FatJet_eta'].array()[n_fatjet][:,1]) < 1.5
RT = ifile['MET'].array()[n_fatjet] / ifile['MT'].array()[n_fatjet] > 0.15
DeltaPhimin = np.minimum(abs(ifile['FatJet_phi'].array()[n_fatjet][:,0] - ifile['MET_phi'].array()[n_fatjet][:]), abs(ifile['FatJet_phi'].array()[n_fatjet][:,1] - ifile['MET_phi'].array()[n_fatjet][:])) < 0.8
nconst1 = ifile['FatJet_nconst'].array()[n_fatjet][:,0] >= 5
nconst2 = ifile['FatJet_nconst'].array()[n_fatjet][:,1] >= 5
sel_mask =  ht & MT & DeltaEta & RT & DeltaPhimin & nconst1 & nconst2
events = ifile.arrays()[n_fatjet][sel_mask]

    
data = {}
for b in branches:
    print(events[b])
    data[b] = events[b]
data['evtweight'] = np.reshape(data['evtweight'], (len(data['evtweight']),-1)) #to get same shape as other inputs (2D array) 

#run on background first to get sum of weights
if(issignal == 0):
    print("# of events after cuts: ", np.shape(data.get("evtweight")))
    print("sum of weights: ", np.sum(data["evtweight"]))
    
#change signal weights to balance signal and background (identical sum of weights for both after cuts)
if(issignal == 1):
    print("# of events after cuts: ", np.shape(data.get("evtweight")))
    print("old weights: ", data.get("evtweight"))
    print("old sum of weights: ", np.sum(data.get("evtweight")))
    #data["evtweight"] = data["evtweight"] * 183073.0/np.sum(data["evtweight"]) #enter sum of signal weights after cuts here
    data["evtweight"] = data["evtweight"] * 45.6636945/np.sum(data["evtweight"]) #enter sum of background weights after cuts here
    print("new weights: ", data.get("evtweight"))
    print("new sum of weights: ", np.sum(data.get("evtweight")))
    
concat_data = []
if maxevents == -1:
    which_indices = [i for i in range(len(data[branches[0]]))]
    random.shuffle(which_indices)#might not be needed
else:
    #which_indices = list(random.sample([i for i in range(len(data[branches[0]]))],maxevents))
    which_indices = [i for i in range(len(data[branches[0]]))]
    random.shuffle(which_indices)
    #print(which_indices)
    
for i in tqdm.tqdm(range(len(which_indices))):
    #print("New event",i)
    if len(data[branches[0]][which_indices[i]]) == 0:
        continue
    evt_data = []
    for b in branches:
        #print(b,data[b][i])
        if len(evt_data) == 0:
            evt_data = np.expand_dims(padarray(np.array(data[b][which_indices[i]]),100),axis=1)#.resize(100)
        else:
            evt_data = np.concatenate((evt_data,np.expand_dims(padarray(np.array(data[b][which_indices[i]]),100),axis=1)),axis=-1)
        #print(evt_data)

    concat_data.append(evt_data)

    if len(concat_data) == maxevents:
        break

concat_data = np.array(concat_data)
target = np.array([1 if issignal else 0 for e in range(len(concat_data))])

print(concat_data.shape)
print(target.shape)
print(target)
#print(arrs)

split1 = round(0.4 * len(concat_data))
split2 = round(0.6 * len(concat_data))
print(split1, split2)


for t in ['train','val','test']:
    hf = h5py.File(infile.replace(".root","_%s_all_features.h5"%t), 'w')
    if t=='train':
        hf.create_dataset('features', data=concat_data[:split1,:,:])
        hf.create_dataset('target', data=target[:split1])
    if t=='val':
        hf.create_dataset('features', data=concat_data[split1:split2,:,:])
        hf.create_dataset('target', data=target[split1:split2])
    if t=='test':
        hf.create_dataset('features', data=concat_data[split2:,:,:])
        hf.create_dataset('target', data=target[split2:])
    hf.close()


