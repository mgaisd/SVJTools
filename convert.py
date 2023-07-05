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

gen_branches = ['GenJetConst_pt','GenJetConst_eta','GenJetConst_phi','GenJetConst_mass']
scouting_branches = ['FatJetConst_pt','FatJetConst_eta','FatJetConst_phi','FatJetConst_mass']
offline_branches = ['RecoJetConst_pt','RecoJetConst_eta','RecoJetConst_phi','RecoJetConst_mass']

ifile = uproot.open(infile)
events = ifile['Events']
#events.show()

if datatype == 'gen':
    branches = gen_branches
elif datatype == 'scouting':
    branches = scouting_branches
elif datatype == 'offline':
    branches = offline_branches
    
data = {}
for b in branches:
    print(events[b].array())
    data[b] = events[b].array()
    
concat_data = []
if maxevents == -1:
    which_indices = [i for i in range(len(data[branches[0]]))]
    random.shuffle(which_indices)#might not be needed
else:
    #which_indices = list(random.sample([i for i in range(len(data[branches[0]]))],maxevents))
    which_indices = [i for i in range(len(data[branches[0]]))]
    random.shuffle(which_indices)
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

split1 = round(0.4 * len(concat_data))
split2 = round(0.6 * len(concat_data))
print(split1, split2)


for t in ['train','val','test']:
    hf = h5py.File(infile.replace(".root","_%s.h5"%t), 'w')
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


