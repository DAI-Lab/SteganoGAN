#!/bin/bash
rm -r results/*

for data_depth in `seq 1 6`; do
    python train.py --dataset div2k --architecture basic --data_depth $data_depth
    python train.py --dataset div2k --architecture residual --data_depth $data_depth
    python train.py --dataset div2k --architecture dense --data_depth $data_depth
done  

for data_depth in `seq 1 6`; do
    python train.py --dataset mscoco --architecture basic --data_depth $data_depth
    python train.py --dataset mscoco --architecture residual --data_depth $data_depth
    python train.py --dataset mscoco --architecture dense --data_depth $data_depth
done  
