#!/bin/sh

random=${RANDOM}

common_params="path=default data=clotho gpus=1 data.resample_rate=32000 data.bsize=8 epochs=50"

./run.sh train ${common_params} tag="${random}_CLOTHO_LAT_greedy" expt=lat audio=meldb expt.beam_size=1
./run.sh train ${common_params} tag="${random}_CLOTHO_LAT_beam10" expt=lat audio=meldb expt.beam_size=10 expt.beam_alpha=0.0
./run.sh train ${common_params} tag="${random}_CLOTHO_CNNT_greedy" expt=cnnt audio=raw expt.beam_size=1
./run.sh train ${common_params} tag="${random}_CLOTHO_CNNT_beam10" expt=cnnt audio=raw expt.beam_size=10 expt.beam_alpha=0.0
