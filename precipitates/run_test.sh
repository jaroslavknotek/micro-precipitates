#!/bin/bash

CS=32
PATIENCE=10
TEST_DIR=../data/test/IN
FILTER=9
TEST_NAME=$1

for d in ../data/202304{1,2}*/ ; do
    dir_name=$(basename $d)
    
    TRAIN_PATH="../data/$dir_name/labeled/"
    
#     python train.py --crop-stride $CS --train-data $TRAIN_PATH --filter-size $FILTER --loss bc  --output "../tmp/$TEST_NAME/$dir_name/filter-$FILTER-bc-$CS" --patience $PATIENCE --test-dir $TEST_DIR
#     python train.py --crop-stride $CS --train-data $TRAIN_PATH --filter-size $FILTER --loss dwbc  --output "../tmp/$TEST_NAME/$dir_name/filter-$FILTER-dwbc-$CS" --patience $PATIENCE --test-dir $TEST_DIR
#     python train.py --crop-stride $CS --train-data $TRAIN_PATH --filter-size $FILTER --loss wbc --wbc-weight-zero 1 --wbc-weight-one 2  --output "../tmp/$TEST_NAME/$dir_name/filter-$FILTER-wbc-1-2-$CS" --patience $PATIENCE --test-dir $TEST_DIR
#     python train.py --crop-stride $CS --train-data $TRAIN_PATH --filter-size $FILTER --loss wbc --wbc-weight-zero 1 --wbc-weight-one 5  --output "../tmp/$TEST_NAME/$dir_name/filter-$FILTER-wbc-1-5-$CS" --patience $PATIENCE --test-dir $TEST_DIR
    python train.py --crop-stride $CS --train-data $TRAIN_PATH --filter-size $FILTER --loss wbc --wbc-weight-zero 2 --wbc-weight-one 1  --output "../tmp/$TEST_NAME/$dir_name/filter-$FILTER-wbc-2-1-$CS" --patience $PATIENCE --test-dir $TEST_DIR

done

