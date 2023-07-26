import ROOT
import sys


iFile = ROOT.TFile.Open(sys.argv[2],"READ")
print(iFile.Get("Events").GetEntries())
number_of_events = float(iFile.Get("Events").GetEntries())
iFile.Close()

oFile = ROOT.TFile.Open(sys.argv[1],"UPDATE")
new_number_of_events = float(oFile.Get("Events").GetEntries())
h = ROOT.TH1F("normalization", "normalization",1, 0, 1)
h.SetBinContent(1,number_of_events)
h.Write()
ratio = new_number_of_events / number_of_events
h2 = ROOT.TH1F("event_ratio", "event_ratio",1, 0, 1)
h2.SetBinContent(1,ratio)
h2.Write()

oFile.Close()
