#!/bin/bash
rm -r results/*

for data_depth in `seq 1 3`; do
    echo "Datadepth $data_depth"
    python train.py --dataset div2k --architecture basic --data_depth $data_depth --epoch  3
    # python train.py --dataset div2k --architecture residual --data_depth $data_depth
    # python train.py --dataset div2k --architecture dense --data_depth $data_depth
done
