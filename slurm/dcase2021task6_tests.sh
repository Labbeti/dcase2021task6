#!/bin/sh

random=$RANDOM
common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 epochs=50"

# ./run.sh train ${common_params} expt=lat audio=meldb tag="${random}_CLOLAT_32K_meldb_bsize8" data.bsize=8
# ./run.sh train ${common_params} expt=st audio=meldb tag="${random}_CLOST_32K_meldb_bsize8" data.bsize=8
# ./run.sh train ${common_params} expt=cst audio=cmeldb tag="${random}_CLOCST_32K_cmeldb_bsize4" data.bsize=4
# ./run.sh train ${common_params} expt=convlstm audio=raw tag="${random}_CLOCL_32K_raw_bsize8" data.bsize=8


common_params="path=osirim expt=convlstm audio=raw data=clotho gpus=1 data.resample_rate=32000 epochs=50 data.bsize=8"

# ./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize8" expt.freeze_cnn14=false expt.beam_size=10
# ./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize8_greedy" expt.freeze_cnn14=false expt.beam_size=1
# ./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize8_freeze" expt.freeze_cnn14=true expt.beam_size=10
# ./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize8_greedy_freeze" expt.freeze_cnn14=true expt.beam_size=1

#./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize8_greedy_3augm" expt.beam_size=1 expt.use_cutout=1 expt.use_mixup=1 expt.use_specaugm=1

common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 epochs=50 data.bsize=8 expt=lat audio=meldb"

#./run.sh train ${common_params} tag="${random}_CLOLAT_32K_meldb_bsize8_greedy" expt.beam_size=1
#./run.sh train ${common_params} tag="${random}_CLOLAT_32K_meldb_bsize8_greedy_mixup" expt.beam_size=1 expt.use_mixup=1

common_params="path=osirim data=merge gpus=1 data.resample_rate=32000 epochs=50 expt.beam_size=1 data.bsize=16"
# ./run.sh train ${common_params} tag="${random}_MERLAT_32K_meldb_bsize16_greedy" expt=lat audio=meldb
# ./run.sh train ${common_params} tag="${random}_MERCL_32K_raw_bsize16_greedy" expt=convlstm audio=raw
# ./run.sh train ${common_params} tag="${random}_MERCL_32K_raw_bsize16_greedy_cutout" expt=convlstm audio=raw expt.use_cutout=1
# ./run.sh train ${common_params} tag="${random}_MERCL_32K_raw_bsize16_greedy_specaugm" expt=convlstm audio=raw expt.use_specaugm=1

common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 epochs=50 expt.beam_size=1 data.bsize=16"
# ./run.sh train ${common_params} tag="${random}_CLOLAT_32K_meldb_bsize16_greedy" expt=lat audio=meldb
# ./run.sh train ${common_params} tag="${random}_CLOCL_32K_raw_bsize16_greedy" expt=convlstm audio=raw


# ----- TEST ----------------------------------------------------------------------------------------------------------------------------------------------------
common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 data.bsize=8"
resume='/users/samova/elabbe/root_aac/AAC/logs/Clotho/train_ConvLSTM/2021-06-14_00:28:06_5691_CLOCL_32K_raw_bsize8_greedy_freeze/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize8_beam01" expt=convlstm audio=raw resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize8_beam10" expt=convlstm audio=raw resume="${resume}" expt.beam_size=10
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize8_beam20" expt=convlstm audio=raw resume="${resume}" expt.beam_size=20
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize8_beam10_alpha0" expt=convlstm audio=raw resume="${resume}" expt.beam_size=10 expt.beam_alpha=0.0

common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 data.bsize=16"
resume='/users/samova/elabbe/root_aac/AAC/logs/Clotho/train_ConvLSTM/2021-06-15_10:44:58_31564_CLOCL_32K_raw_bsize16_greedy/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize16_beam01" expt=convlstm audio=raw resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize16_beam10" expt=convlstm audio=raw resume="${resume}" expt.beam_size=10
# ./run.sh test ${common_params} tag="${random}_CLOCL_32K_bsize16_beam20" expt=convlstm audio=raw resume="${resume}" expt.beam_size=20

common_params="path=osirim data=merge gpus=1 data.resample_rate=32000 data.bsize=16"
resume='/users/samova/elabbe/root_aac/AAC/logs/MergeClotho/train_ConvLSTM/2021-06-14_23:04:34_28105_MERCL_32K_raw_bsize16_greedy/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_MERCL_32K_bsize16_beam01" expt=convlstm audio=raw resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_MERCL_32K_bsize16_beam10" expt=convlstm audio=raw resume="${resume}" expt.beam_size=10
# ./run.sh test ${common_params} tag="${random}_MERCL_32K_bsize16_beam20" expt=convlstm audio=raw resume="${resume}" expt.beam_size=20


common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 data.bsize=8"
resume='/users/samova/elabbe/root_aac/AAC/logs/Clotho/train_ListenAttendTell/2021-06-14_13:01:32_4801_CLOLAT_32K_meldb_bsize8_greedy/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_CLOLAT_32K_bsize8_beam01" expt=lat audio=meldb resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_CLOLAT_32K_bsize8_beam10" expt=lat audio=meldb resume="${resume}" expt.beam_size=10
# ./run.sh test ${common_params} tag="${random}_CLOLAT_32K_bsize8_beam10_alpha0" expt=lat audio=meldb resume="${resume}" expt.beam_size=10 expt.beam_alpha=0.0

common_params="path=osirim data=clotho gpus=1 data.resample_rate=32000 data.bsize=16"
resume='/users/samova/elabbe/root_aac/AAC/logs/Clotho/train_ListenAttendTell/2021-06-14_20:13:35_25525_CLOLAT_32K_meldb_bsize16_greedy/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_CLOLAT_32K_bsize16_beam01" expt=lat audio=meldb resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_CLOLAT_32K_bsize16_beam10" expt=lat audio=meldb resume="${resume}" expt.beam_size=10

common_params="path=osirim data=merge gpus=1 data.resample_rate=32000 data.bsize=16"
resume='/users/samova/elabbe/root_aac/AAC/logs/MergeClotho/train_ListenAttendTell/2021-06-14_23:04:33_28105_MERLAT_32K_meldb_bsize16_greedy/checkpoints/best.ckpt'
# ./run.sh test ${common_params} tag="${random}_MERLAT_32K_bsize16_beam01" expt=lat audio=meldb resume="${resume}" expt.beam_size=1
# ./run.sh test ${common_params} tag="${random}_MERLAT_32K_bsize16_beam10" expt=lat audio=meldb resume="${resume}" expt.beam_size=10
