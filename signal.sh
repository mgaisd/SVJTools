#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=slc7_amd64_gcc700

### GEN                                                                                                                                                                                                   

cp /work/mgais/svj/signal/input/$1.root .
 
echo "###########"
echo "GEN step"
echo "###########"
scram p CMSSW CMSSW_10_6_26
cd CMSSW_10_6_26/src
tar xvaf ../../cmssw.tar.gz
eval `scram runtime -sh`
scram b -j 1
cd ../../

cp CMSSW_10_6_26/src/PhysicsTools/SUEPScouting/test/ScoutingNanoAOD_cfg.py .

cmsRun ScoutingNanoAOD_cfg.py inputFiles=file:$1.root outputFile=scouting_$1.root maxEvents=1000 isMC=true era=2018

cp scouting_$1_numEvent1000.root /work/mgais/svj/signal/outfiles/

