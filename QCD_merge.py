import uproot
import pandas as pd
import glob
import ROOT
import random
import awkward as ak
import numpy as np
import sys
import subprocess

#if "/functions.so" not in ROOT.gSystem.GetLibraries():
ROOT.gSystem.CompileMacro("functions.cc","k")


#first determine QCD bin with lowest lumi = number of events/crosssection  
#take all events of that bin
#randomly take n=minlumi*crosssection events of all other bins
#maybe create a list of appropriate number of 0s/1s and randomly shuffle it, then use Filter() to throw out events
#write all events into single QCD output file with max 100GB size (see rdataframe: define to add column, snapshot to write the new file)

#add option to either save scouting, gen or offline


filepath = "/ceph/mgais/svj/background_pt200/"


#calculate lumis for all QCD bins
data = pd.read_csv('merge_crosssections.txt', sep=",")
#print(data)
#print (data.columns.values[0])

lumi = []
N = []

for i in range(0, 6):
    ifile = ROOT.TFile.Open(filepath + str(data.columns.values[i]),"READ")

    crosssection = data.iloc[0,i]
    N_events = ifile.Get("normalization").GetBinContent(1)
    normalizedLumi =  N_events / crosssection
    #print(ifile.Get("normalization").GetBinContent(1))
    N.append("%.0f"%N_events)
    lumi.append(float(normalizedLumi))
    ifile.Close()


data.loc[len(data)] = lumi
data.loc[len(data)] = N
print(data)

#determine lowest lumi
min = data.iloc[1].min()
#print(data.iloc[1])
#print(min)
min_bin = data.columns.values[(data == min).iloc[1]]
print("bin with lowest luminosity",str(min_bin))


#run on all files
for i in range(0, 6):
    print("starting with file ", i+1)
    df = ROOT.RDataFrame("mmtree/tree", filepath + str(data.columns.values[i]))
    ifile = ROOT.TFile.Open(filepath + str(data.columns.values[i]),"READ")

    #get list of branches
    branches = []
    for branch in ifile.mmtree.tree.GetListOfBranches():
        name = branch.GetName()
        #print("Branch: %s" %name)                                                                                                                                                                        
        branches.append(str(name))
    branches.remove('hltResultName')

    #remove events
    if (str(data.columns.values[i]) != min_bin):
        print("--------------")
        print("lowest lumi: ",float(min))
        print("crossection: ",data.iloc[0, i])
        print("N_events: ",float(data.iloc[2, i]))
        ratio = float(min) * float(data.iloc[0, i]) / float(data.iloc[2, i])
        print("ratio: ",ratio)
        new_N = float(min) * float(data.iloc[0, i])
        print("new N_events: ",new_N)
        
        df= df.Define("mask", "randomSelection2(%f)"%ratio)
        df= df.Filter("mask == 1")

    #remove branches
    #if "scouting" in sys.argv[1]:
    if (sys.argv[1] == "scouting"):
        branches.remove('n_genjet')
        branches.remove('GenJet_pt')
        branches.remove('GenJet_eta')
        branches.remove('GenJet_phi')
        branches.remove('GenJet_mass')
        branches.remove('GenJetConst_pt')
        branches.remove('GenJetConst_eta')
        branches.remove('GenJetConst_phi')
        branches.remove('GenJetConst_mass')
        branches.remove('GenJetConst_pdgID')
        branches.remove('GenJetConst_charge')
        
        branches.remove('n_recojet')
        branches.remove('RecoJet_pt')
        branches.remove('RecoJet_eta')
        branches.remove('RecoJet_phi')
        branches.remove('RecoJet_mass')
        branches.remove('RecoJetConst_pt')
        branches.remove('RecoJetConst_eta')
        branches.remove('RecoJetConst_phi')
        branches.remove('RecoJetConst_mass')
        branches.remove('RecoJetConst_pdgID')
        branches.remove('RecoJetConst_charge')

        
    if (sys.argv[1] == "gen"):
        branches.remove('n_fatjet')
        branches.remove('FatJet_pt')
        branches.remove('FatJet_eta')
        branches.remove('FatJet_phi')
        branches.remove('FatJet_mass')
        branches.remove('FatJet_area')
        branches.remove('FatJet_n2b1')
        branches.remove('FatJet_n3b1')
        branches.remove('FatJet_tau1')
        branches.remove('FatJet_tau2')
        branches.remove('FatJet_tau3')
        branches.remove('FatJet_tau4')
        branches.remove('FatJet_tau21')
        branches.remove('FatJet_tau32')
        branches.remove('FatJet_msoftdrop')
        branches.remove('FatJet_mtrim')
        branches.remove('FatJet_nconst')
        branches.remove('FatJet_girth')
            
        branches.remove('FatJet_sj1_pt')
        branches.remove('FatJet_sj1_eta')
        branches.remove('FatJet_sj1_phi')
        branches.remove('FatJet_sj1_mass')
        branches.remove('FatJet_sj2_pt')
        branches.remove('FatJet_sj2_eta')
        branches.remove('FatJet_sj2_phi')
        branches.remove('FatJet_sj2_mass')
        
        branches.remove('FatJetConst_pt')
        branches.remove('FatJetConst_eta')
        branches.remove('FatJetConst_phi')
        branches.remove('FatJetConst_mass')
        branches.remove('FatJetConst_pdgID')
        branches.remove('FatJetConst_charge')
        
        branches.remove('n_fatjet_CA')
        branches.remove('FatJet_pt_CA')
        branches.remove('FatJet_eta_CA')
        branches.remove('FatJet_phi_CA')
        branches.remove('FatJet_mass_CA')
        branches.remove('FatJet_area_CA')
        branches.remove('FatJet_n2b1_CA')
        branches.remove('FatJet_n3b1_CA')
        branches.remove('FatJet_tau1_CA')
        branches.remove('FatJet_tau2_CA')
        branches.remove('FatJet_tau3_CA')
        branches.remove('FatJet_tau4_CA')
        branches.remove('FatJet_tau21_CA')
        branches.remove('FatJet_tau32_CA')
        branches.remove('FatJet_msoftdrop_CA')
        branches.remove('FatJet_mtrim_CA')
        branches.remove('FatJet_nconst_CA')
        branches.remove('FatJet_girth_CA')
        
        branches.remove('FatJet_sj1_pt_CA')
        branches.remove('FatJet_sj1_eta_CA')
        branches.remove('FatJet_sj1_phi_CA')
        branches.remove('FatJet_sj1_mass_CA')
        branches.remove('FatJet_sj2_pt_CA')
        branches.remove('FatJet_sj2_eta_CA')
        branches.remove('FatJet_sj2_phi_CA')
        branches.remove('FatJet_sj2_mass_CA')

        branches.remove('n_recojet')
        branches.remove('RecoJet_pt')
        branches.remove('RecoJet_eta')
        branches.remove('RecoJet_phi')
        branches.remove('RecoJet_mass')
        branches.remove('RecoJetConst_pt')
        branches.remove('RecoJetConst_eta')
        branches.remove('RecoJetConst_phi')
        branches.remove('RecoJetConst_mass')
        branches.remove('RecoJetConst_pdgID')
        branches.remove('RecoJetConst_charge')

            
    if (sys.argv[1] == "offline"):
        branches.remove('n_fatjet')
        branches.remove('FatJet_pt')
        branches.remove('FatJet_eta')
        branches.remove('FatJet_phi')
        branches.remove('FatJet_mass')
        branches.remove('FatJet_area')
        branches.remove('FatJet_n2b1')
        branches.remove('FatJet_n3b1')
        branches.remove('FatJet_tau1')
        branches.remove('FatJet_tau2')
        branches.remove('FatJet_tau3')
        branches.remove('FatJet_tau4')
        branches.remove('FatJet_tau21')
        branches.remove('FatJet_tau32')
        branches.remove('FatJet_msoftdrop')
        branches.remove('FatJet_mtrim')
        branches.remove('FatJet_nconst')
        branches.remove('FatJet_girth')
            
        branches.remove('FatJet_sj1_pt')
        branches.remove('FatJet_sj1_eta')
        branches.remove('FatJet_sj1_phi')
        branches.remove('FatJet_sj1_mass')
        branches.remove('FatJet_sj2_pt')
        branches.remove('FatJet_sj2_eta')
        branches.remove('FatJet_sj2_phi')
        branches.remove('FatJet_sj2_mass')
        
        branches.remove('FatJetConst_pt')
        branches.remove('FatJetConst_eta')
        branches.remove('FatJetConst_phi')
        branches.remove('FatJetConst_mass')
        branches.remove('FatJetConst_pdgID')
        branches.remove('FatJetConst_charge')

        branches.remove('n_fatjet_CA')
        branches.remove('FatJet_pt_CA')
        branches.remove('FatJet_eta_CA')
        branches.remove('FatJet_phi_CA')
        branches.remove('FatJet_mass_CA')
        branches.remove('FatJet_area_CA')
        branches.remove('FatJet_n2b1_CA')
        branches.remove('FatJet_n3b1_CA')
        branches.remove('FatJet_tau1_CA')
        branches.remove('FatJet_tau2_CA')
        branches.remove('FatJet_tau3_CA')
        branches.remove('FatJet_tau4_CA')
        branches.remove('FatJet_tau21_CA')
        branches.remove('FatJet_tau32_CA')
        branches.remove('FatJet_msoftdrop_CA')
        branches.remove('FatJet_mtrim_CA')
        branches.remove('FatJet_nconst_CA')
        branches.remove('FatJet_girth_CA')
            
        branches.remove('FatJet_sj1_pt_CA')
        branches.remove('FatJet_sj1_eta_CA')
        branches.remove('FatJet_sj1_phi_CA')
        branches.remove('FatJet_sj1_mass_CA')
        branches.remove('FatJet_sj2_pt_CA')
        branches.remove('FatJet_sj2_eta_CA')
        branches.remove('FatJet_sj2_phi_CA')
        branches.remove('FatJet_sj2_mass_CA')
        
        branches.remove('n_genjet')
        branches.remove('GenJet_pt')
        branches.remove('GenJet_eta')
        branches.remove('GenJet_phi')
        branches.remove('GenJet_mass')
        branches.remove('GenJetConst_pt')
        branches.remove('GenJetConst_eta')
        branches.remove('GenJetConst_phi')
        branches.remove('GenJetConst_mass')
        branches.remove('GenJetConst_pdgID')
        branches.remove('GenJetConst_charge')

     
    ofile=filepath + str(data.columns.values[i]).replace(".root","_" + sys.argv[1] + ".root")
    print(ofile)
    #print(branches)
    df.Snapshot("Events", ofile, branches)

outfile = filepath + "QCD_" + sys.argv[1] + ".root"
single_files = filepath + "*_" + sys.argv[1] + ".root"
print(outfile, single_files)
subprocess.call(f"hadd {outfile} {single_files}",shell=True)


        

        
