import h5py
import matplotlib.pyplot as plt
import numpy as np

import sys

sig = sys.argv[1]
bkg = sys.argv[2]
binning = {0:np.linspace(0.1,200,40),1:np.linspace(-2.5,2.5,40),2:np.linspace(-3.1415,3.1415,40),3:np.linspace(-0.1,10,40)}

for i in range(4):
    fig,ax=plt.subplots()
    with h5py.File(sig, "r") as s:
        feat = s['features'][()]
        flat_pt = feat[:,:,i].flatten()
        flat_pt = flat_pt[feat[:,:,0].flatten()!=0]
        plt.hist(flat_pt,label='sig',bins=binning[i],density=True,histtype='step')
    with h5py.File(bkg, "r") as b:
        feat = b['features'][()]
        flat_pt = feat[:,:,i].flatten()
        flat_pt = flat_pt[feat[:,:,0].flatten()!=0]
        plt.hist(flat_pt,label='bkg',bins=binning[i],density=True,histtype='step')
    plt.legend()
    plt.savefig("/etpwww/web/mgais/public_html/svj/test_feat_%i.png"%i)


fig,ax=plt.subplots()
with h5py.File(sig, "r") as s:
    feat = s['target'][()]
    plt.hist(feat,label='sig',density=True,histtype='step')
with h5py.File(bkg, "r") as b:
    feat = b['target'][()]
    plt.hist(feat,label='bkg',density=True,histtype='step')
plt.legend()
plt.savefig("/etpwww/web/mgais/public_html/svj/test_label.png")

