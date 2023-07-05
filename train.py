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

    def __init__(self, root, max_events=1e8, datatype='scouting'):
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
            feats = f['features'][idx_in_file,:Npfc,:]
            x_pfc = torch.from_numpy(feats).float()
            x = torch.from_numpy(feats).float()
            y = np.array(f['target'][idx_in_file],dtype='f')
            y = torch.from_numpy(y)
        
        self.processed_events += 1
        #if self.processed_events >= self.max_events:
        #    return


        #print("New jet")
        #print(x_pfc)
        #print(y)
        
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
            nn.Linear(8, 1),
            nn.ELU()
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

        #out, batch = avg_pool_x(batch_pf, feats3, batch_pf)
        #out = self.output(out)

        batch = batch_pf
        out  = global_add_pool(feats3, batch_pf)
        out = self.output(out)
        
        return out, batch
    
batchsize = 300

trainpath = sys.argv[1]
valpath = trainpath.replace("train/", "val/")
mpath = sys.argv[2]

print('Loading train dataset at', trainpath)
print('Loading val dataset at', valpath)

max_events_train = 1000000 #100000
max_events_val = int(max_events_train*0.5)

data_train = SVJV1(trainpath,max_events=max_events_train,datatype='scouting')
data_val = SVJV1(valpath,max_events=max_events_val,datatype='scouting')

train_loader = DataLoader(data_train, batch_size=batchsize,shuffle=True,
                          follow_batch=['x_pf'])
val_loader = DataLoader(data_val, batch_size=batchsize,shuffle=True,
                         follow_batch=['x_pf'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

network = SVJNet().to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def train():
    network.train()
    counter = 0

    total_loss = 0
    for data in tqdm.tqdm(train_loader):
        counter += 1

        data = data.to(device)
        optimizer.zero_grad()
        out = network(data.x_pf,
                    data.x_pf_batch)

        #print(data.x_pf_batch)
        #print(np.bincount(data.x_pf_batch.cpu().detach().numpy()))
        #print(data.x_pf[0])
        #print(data.x_pf[1])
        loss = nn.BCEWithLogitsLoss(reduction='sum')(out[0].view(-1),data.y.float())
        #print(torch.sigmoid(out[0].view(-1)))
        #print(data.y.float())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if counter*batchsize > max_events_train:
            break
        
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def validate():
    network.eval()
    total_loss = 0
    counter = 0
    for data in tqdm.tqdm(val_loader):
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')                                                                                                           
        data = data.to(device)
        with torch.no_grad():
            out = network(data.x_pf,
                       data.x_pf_batch)
 
            loss = nn.BCEWithLogitsLoss(reduction='sum')(out[0].view(-1),data.y.float())

            total_loss += loss.item()
            if  counter*batchsize > max_events_train:
                break
    return total_loss / len(val_loader.dataset)

best_val_loss = 1e11
all_train_loss = []
all_val_loss = []
loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, 100):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} jets')
    loss = train()
    scheduler.step()

    print(f'Validating Epoch {epoch} on {len(val_loader.dataset)} jets')
    loss_val = validate()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    all_train_loss.append(loss)
    all_val_loss.append(loss_val)
    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)

    if not os.path.exists(mpath):
        subprocess.call("mkdir -p %s"%mpath,shell=True)

    state_dicts = {'model':network.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()}
    df.to_csv("%s/"%mpath+"/loss.csv")
    torch.save(state_dicts, os.path.join(mpath, f'epoch-{epoch}.pt'))
    
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        torch.save(state_dicts, os.path.join(mpath, 'best-epoch.pt'))

