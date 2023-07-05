import numpy as np
import uproot
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



s_tree=uproot.open("scouting_model/output.root")['tree']
s_branches = s_tree.arrays()
g_tree=uproot.open("gen_model/output.root")['tree']
g_branches = g_tree.arrays()
o_tree=uproot.open("offline_model/output.root")['tree']
o_branches = o_tree.arrays()


s_auc = round(roc_auc_score(s_branches['true'], s_branches['pred']), 3)
s_fpr, s_tpr, _ = roc_curve(s_branches['true'], s_branches['pred'])
g_auc = round(roc_auc_score(g_branches['true'], g_branches['pred']), 3)
g_fpr, g_tpr, _ = roc_curve(g_branches['true'], g_branches['pred'])
o_auc = round(roc_auc_score(o_branches['true'], o_branches['pred']), 3)
o_fpr, o_tpr, _ = roc_curve(o_branches['true'], o_branches['pred'])


fig,ax = plt.subplots(figsize=(7,6))
plt.plot(s_fpr, s_tpr, label="scouting, auc="+str(s_auc))
plt.plot(o_fpr, o_tpr, label="offline, auc="+str(o_auc))
plt.plot(g_fpr, g_tpr, label="gen, auc="+str(g_auc))
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/ROC.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/ROC.pdf',bbox_inches='tight')
