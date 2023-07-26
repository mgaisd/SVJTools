import uproot 
import ROOT
import sys
import numpy as np
import pandas as pd
import os

drop = ['hltResultName', 'genModel']

infile = sys.argv[1] 

j1 = pd.read_csv('scouting1.csv')
#print(df1)
j2 = pd.read_csv('scouting2.csv')

events = uproot.concatenate([infile+":mmtree/tree"])

print(events)
print(len(events['run']))

j1_score = []
j2_score = []
for i,e in enumerate(events['event']):
  try:
    j1_score.append(j1[j1['event'] == e]['pred'].values[0])  
  except:
    j1_score.append(-99)
  try:
    j2_score.append(j2[j2['event'] == e]['pred'].values[0])  
  except:
    j2_score.append(-99)

events['j1_score'] = j1_score
events['j2_score'] = j2_score

n_fatjet_mask = events['n_fatjet'] > 1
events = events[n_fatjet_mask]

oevents = {}
for f in events[0].fields:
  if f in drop:
    continue
  if ("Const" in f) & (f != 'Jet_nConstituents'):
    continue
  if type(events[f][0]) == str:
    print("XXXX")
    print(f)
    print("XXXX")
  oevents[f] = events[f]
print(oevents['j1_score'])
print(type(oevents['j1_score']))

ofile = uproot.recreate("out.root", compression=uproot.ZLIB(4))
ofile['events'] = oevents
