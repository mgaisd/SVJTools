import sys
import numpy as np
import subprocess
import tqdm
import pandas as pd
import os
import os.path as osp
import glob
import h5py
import uproot
import torch
import awkward as ak
import random
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import Sequential, Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter

def global_add_pool(x, batch, size=None):
    """                                                                                                                                                                                                    
    Globally pool node embeddings into graph embeddings, via elementwise sum.                                                                                                                              
    Pooling function takes in node embedding [num_nodes x emb_dim] and                                                                                                                                     
    batch (indices) and outputs graph embedding [num_graphs x emb_dim].                                                                                                                                    
                                                                                                                                                                                                           
    Args:                                                                                                                                                                                                  
        x (torch.tensor): Input node embeddings                                                                                                                                                            
        batch (torch.tensor): Batch tensor that indicates which node                                                                                                                                       
        belongs to which graph                                                                                                                                                                             
        size (optional): Total number of graphs. Can be auto-inferred.                                                                                                                                     
                                                                                                                                                                                                           
    Returns: Pooled graph embeddings                                                                                                                                                                       
                                                                                                                                                                                                           
    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')



class SVJV1(Dataset):
    '''                                                                                                                               

        input: particle (candidates)

    '''

    url = '/dummy/'

    def __init__(self, root, max_events=1e8, datatype='offline'):
        super(SVJV1, self).__init__(root)
        self.processed_events = 0
        self.max_events = max_events
        self.strides = [0]
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(len(f['features'][()]))

        self.strides = np.cumsum(self.strides)
        
    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        #print(file_idx)
        idx_in_file = idx - self.strides[max(0, file_idx)] - 1
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx]) as f:
            Npfc = (f['features'][idx_in_file][:,0] != 0).sum()
            x_pfc = f['features'][idx_in_file,:Npfc,:]
            x_pfc = torch.from_numpy(x_pfc)
            x = x_pfc
            y = np.array(f['target'][idx_in_file],dtype='f')
            y = torch.from_numpy(y)
        
        self.processed_events += 1
        #if self.processed_events >= self.max_events:
        #    return

        return Data(x=x, edge_index=edge_index, y=y, x_pf=x_pfc)

        
class SVJNet(nn.Module):
    def __init__(self):
        super(SVJNet, self).__init__()
        
        hidden_dim = 64
        
        self.pf_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )

        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1)
        )
        
    def forward(self,
                x_pf,
                batch_pf):
        #print(x_ts)
        #x_pf = BatchNorm(x_pf)
        
        x_pf_enc = self.pf_encode(x_pf)
        
        feats1 = self.conv1(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))
        feats2 = self.conv2(x=(feats1, feats1), batch=(batch_pf, batch_pf))
        feats3 = self.conv3(x=(feats2, feats2), batch=(batch_pf, batch_pf))

        batch = batch_pf
        out  = global_add_pool(feats3, batch_pf)
        out = self.output(out)

        return out, batch
    
batchsize = 1

testpath = sys.argv[1]
mpath = sys.argv[2]

print('Loading test dataset at', testpath)


max_events_train = 1000000 #100000
max_events_val = int(max_events_train*0.5)

data_test = SVJV1(testpath,max_events=max_events_train,datatype='offline')

test_loader = DataLoader(data_test, batch_size=batchsize,shuffle=True,
                          follow_batch=['x_pf'])

#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

network = SVJNet().to(device)
network.load_state_dict(torch.load(mpath+"/best-epoch.pt", map_location=torch.device('cpu'))['model'])

@torch.no_grad()
def test():
    network.eval()
    total_loss = 0
    counter = 0

    #outputarray = np.empty([0,2])
    all_true = []
    all_pred = []
    
    for data in tqdm.tqdm(test_loader):
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')                                                                                                           
        data = data.to(device)

        with torch.no_grad():
            out = network(data.x_pf,
                       data.x_pf_batch)
 
            a = torch.sigmoid(out[0].view(-1)).cpu().item()
            b = data.y.float().cpu().item()
            #print(a)
            #print(b)
            all_true.append(b)
            all_pred.append(a)
            #if counter > 400:
             #   break
            
    all_data = {'true':all_true,'pred':all_pred}
    #df = pd.DataFrame.from_dict(all_data)
    #df.to_csv("%s/"%mpath+"/infer.csv",index=False)
    ofile = uproot.recreate("%s/"%mpath+"/output.root")
    ofile['tree'] = all_data
    
    return total_loss / len(test_loader.dataset)

loss_test = test()


#infer plot
tree = uproot.open("%s/"%mpath+"/output.root")['tree']
branches = tree.arrays()
sig_mask = branches['true'] == 1
bg_mask = branches['true'] == 0

print(branches['pred'][sig_mask])
fig,ax = plt.subplots(figsize=(7,6))
bins=np.linspace(0,1,30)
plt.hist(branches['pred'][sig_mask],histtype='step',density=True,label='signal',bins=bins)
plt.hist(branches['pred'][bg_mask],histtype='step',density=True,label='background',bins=bins)
plt.xlabel("gnn output")
plt.ylabel("Norm. to unit area")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
if "scouting" in testpath:
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_scouting.png',bbox_inches='tight',dpi=300)
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_scouting.pdf',bbox_inches='tight')
if "gen" in testpath:
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_gen.png',bbox_inches='tight',dpi=300)
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_gen.pdf',bbox_inches='tight')
if "offline" in testpath:
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_offline.png',bbox_inches='tight',dpi=300)
    fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/infer_offline.pdf',bbox_inches='tight')

