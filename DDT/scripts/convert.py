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
datatype = sys.argv[2]
issignal = bool(int(sys.argv[3]))
try:
    maxevents = int(sys.argv[4]) ### also means random shuffling
except:
    maxevents = -1

#maxevents = int(sys.argv[4])

scouting1_branches = ['FatJet1Const_pt','FatJet1Const_eta','FatJet1Const_phi','FatJet1Const_mass','event']
scouting2_branches = ['FatJet2Const_pt','FatJet2Const_eta','FatJet2Const_phi','FatJet2Const_mass','event']

ifile = uproot.open(infile)
branch = ifile['mmtree/tree'].arrays()
ht_mask = branch['ht'] > 500
events = branch[ht_mask]
#events = ifile['Events'].arrays()
#print(branch['ht'])
#print("-------------------------------------------")
#print(events['ht'])
#events.show()

if datatype == 'scouting1':
    branches = scouting1_branches
elif datatype == 'scouting2':
    branches = scouting2_branches
    
data = {}
for b in branches:
    print(events[b])
    data[b] = events[b]
data['event'] = np.reshape(data['event'], (len(data['event']),1))

concat_data = []
if maxevents == -1:
    which_indices = [i for i in range(len(data[branches[0]]))] #dont want random shuffling here
    #random.shuffle(which_indices)#might not be needed
else:
    #which_indices = list(random.sample([i for i in range(len(data[branches[0]]))],maxevents))
    which_indices = [i for i in range(len(data[branches[0]]))]
    #random.shuffle(which_indices)
    print(which_indices)
    
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
    #print
    #arrs = [np.stack(events[b].array(), axis=0) for b in gen_branches]

print(concat_data.shape)
print(target.shape)
print(target)
#print(arrs)

hf = h5py.File(infile.replace(".root", str(datatype) + ".h5"), 'w')
hf.create_dataset('features', data=concat_data[:,:,:])
hf.create_dataset('target', data=target[:])
hf.close()


