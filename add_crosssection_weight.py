import pandas as pd
import ROOT
import sys
import subprocess


data = pd.read_csv('crosssections.txt', sep=",") 
print(data)

crosssection = data[sys.argv[1].split("/")[-1]]



ifile = ROOT.TFile.Open(sys.argv[1],"READ")


if "signal" in sys.argv[1]:
    dummy = 1.0
    weight= "%.8f"%dummy
else:
    N_events = ifile.Get("normalization").GetBinContent(1)
    normalizedWeight = crosssection/N_events
    print(ifile.Get("normalization").GetBinContent(1))
    weight="%.8f"%normalizedWeight
 
print(weight)
print(crosssection)

branches = []
#print("These are all the columns available to this dataframe:")
for branch in ifile.mmtree.tree.GetListOfBranches():
    name = branch.GetName()
    #print("Branch: %s" %name)
    branches.append(str(name))
branches.remove('hltResultName')
#print(branches)

ifile.Close()

ofile=sys.argv[-1].replace(".root","_final.root")
df = ROOT.RDataFrame("mmtree/tree", sys.argv[1])
df = df.Define("evtweight",weight)
#df = df.Define("evtweight","1.4")
branches.append("evtweight")

df.Snapshot("Events", ofile, branches)

subprocess.call(f"mv {ofile} {sys.argv[-1]}",shell=True)
