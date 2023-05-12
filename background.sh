#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=slc7_amd64_gcc700

### GEN                                                                                                                                                                                                   

cp /ceph/bmaier/svj/background/$1.root .
 
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
cp /work/mgais/svj/add_normalization.py .

cmsRun ScoutingNanoAOD_cfg.py inputFiles=file:$1.root outputFile=output_$1.root maxEvents=-1 isMC=true era=2018

scram p CMSSW CMSSW_12_5_0
cd CMSSW_12_5_0/src
eval `scram runtime -sh`
cd ../../

python3 add_normalization.py output_$1.root $1.root

cp output_$1.root /ceph/mgais/svj/backgroundv2/outfiles/

