#!/bin/sh

currentDIR="$( pwd )"
delphesDIR=$currentDIR/MG5/Delphes
pythiaDIR=$currentDIR/MG5/HEPTools/pythia8
lhapdfDIR=$currentDIR/MG5/HEPTools/lhapdf6_py3
#Make sure pythia can be found by Delphes
export LD_LIBRARY_PATH=$pythiaDIR/lib:$lhapdfDIR/lib:$LD_LIBRARY_PATH
#Make sure Delphes can be found by ROOT
export ROOT_INCLUDE_PATH=$delphesDIR/external
export PYTHON_PATH=$lhapdfDIR/lib:$PYTHON_PATH

