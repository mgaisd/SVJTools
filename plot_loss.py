import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

scouting = pd.read_csv("scouting_model/loss.csv", index_col=[0])
print("scouting:")
print(scouting)


fig,ax = plt.subplots(figsize=(7,6))
plt.plot(scouting.index, scouting["train_loss"], label="train_loss")
plt.plot(scouting.index, scouting["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/scouting_loss.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/scouting_loss.pdf',bbox_inches='tight')

gen = pd.read_csv("gen_model/loss.csv", index_col=[0])
print("gen:")
print(gen)


fig,ax = plt.subplots(figsize=(7,6))
plt.plot(gen.index, gen["train_loss"], label="train_loss")
plt.plot(gen.index, gen["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/gen_loss.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/gen_loss.pdf',bbox_inches='tight')

offline = pd.read_csv("offline_model/loss.csv", index_col=[0])
print("offline:")
print(offline)


fig,ax = plt.subplots(figsize=(7,6))
plt.plot(offline.index, offline["train_loss"], label="train_loss")
plt.plot(offline.index, offline["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/offline_loss.png',bbox_inches='tight',dpi=300)
fig.savefig('/etpwww/web/mgais/public_html/svj/figs/dnn/offline_loss.pdf',bbox_inches='tight')

