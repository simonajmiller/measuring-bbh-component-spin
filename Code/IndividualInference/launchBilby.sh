#!/bin/bash

. /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate /home/simona.miller/.conda/envs/bilby_201

job=$1
json=$2
outdir=$3

mkdir -p $outdir

python /home/simona.miller/measuring-bbh-component-spin/Code/IndividualInference/launchBilby.py \
        -job $job \
        -json $json \
        -outdir $outdir

conda deactivate
