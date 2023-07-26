#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=slc7_amd64_gcc700

### GEN                                                                                                                                                                                                   

cp /ceph/bmaier/svj/background/$1 .
 
echo "###########"
echo "GEN step"
echo "###########"
scram p CMSSW CMSSW_10_6_26
cd CMSSW_10_6_26/src
tar xvaf ../../cmssw.tar.gz
eval `scram runtime -sh`
scram b -j 1
cd ../../

tar --strip-components 1 -xvf scripts.tar.gz

cp CMSSW_10_6_26/src/PhysicsTools/SUEPScouting/test/ScoutingNanoAOD_cfg.py .

cmsRun ScoutingNanoAOD_cfg.py inputFiles=file:$1 outputFile=output_$1 saveConst=true onlyScouting=true isMC=true era=2018 #maxEvents=50

#unset python variables
eval `scram unsetenv -sh`

# One setup to rule them all                                                                                                                                                                              
source /cvmfs/belle.cern.ch/tools/b2setup #|| echo "make basf2 work again"
b2setup  release-06-01-11 # || echo "make basf2 work again"
# one path to find them                                                                                                                                                                                   
old_path=$PATH # save PATH to fix venv stuff later                                                                                                                                                        
# one venv to bring them all                                                                                                                                                                              
source /cvmfs/etp.kit.edu/ML/PyG_gnn/eclgravnet_venv/venv/bin/activate
# and in the darkness bind them                                                                                                                                                                           
export PATH=$old_path:/cvmfs/etp.kit.edu/ML/PyG_gnn/eclgravnet_venv/venv/bin
export PYTHONPATH=/cvmfs/etp.kit.edu/ML/PyG_gnn/eclgravnet_venv/venv/lib/python3.8/site-packages/:$PYTHONPATH

#convert.py to create input file for tagger
echo "creating input file"
python3 convert.py output_$1 scouting1 0
python3 convert.py output_$1 scouting2 0
mkdir -p jet1/raw
mkdir -p jet2/raw
mv *scouting1.h5 jet1/raw/
mv *scouting2.h5 jet2/raw/
ls jet1/raw/
ls jet2/raw/

#infer.py to get tagger scores
echo "getting tagger scores"
python3 infer.py ${PWD}/jet1/ /work/mgais/svj/dnn/gnn/HTcut_scouting_model/
python3 infer.py ${PWD}/jet2/ /work/mgais/svj/dnn/gnn/HTcut_scouting_model/

#merge.py to create output file
echo "adding branch to output file"
python3 merge.py output_$1

#need 12.5 for normalization
deactivate

scram p CMSSW CMSSW_12_5_0
cd CMSSW_12_5_0/src
eval `scram runtime -sh`
cd ../../

#add histogram with number of events for normalization
python3 add_normalization.py out.root $1


echo "copying output to ceph"
cp out.root /ceph/mgais/svj/DDT/out_$1

