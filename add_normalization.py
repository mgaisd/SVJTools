import pandas as pd
import ROOT
import sys


iFile = ROOT.TFile.Open(sys.argv[2],"READ")
print(iFile.Get("Events").GetEntries())
number_of_events = float(iFile.Get("Events").GetEntries())
iFile.Close()

oFile = ROOT.TFile.Open(sys.argv[1],"UPDATE")
h = ROOT.TH1F("normalization", "normalization",1, 0, 1)
h.SetBinContent(1,number_of_events)
h.Write()
oFile.Close()
