export CUDA_VISIBLE_DEVICES=0

python train.py --dataset div2k --encoder basic --data_depth 1
python train.py --dataset div2k --encoder basic --data_depth 2
python train.py --dataset div2k --encoder basic --data_depth 3
python train.py --dataset div2k --encoder basic --data_depth 4
python train.py --dataset div2k --encoder basic --data_depth 5
python train.py --dataset div2k --encoder basic --data_depth 6
python train.py --dataset div2k --encoder basic --data_depth 7

python train.py --dataset div2k --encoder residual --data_depth 1
python train.py --dataset div2k --encoder residual --data_depth 2
python train.py --dataset div2k --encoder residual --data_depth 3
python train.py --dataset div2k --encoder residual --data_depth 4
python train.py --dataset div2k --encoder residual --data_depth 5
python train.py --dataset div2k --encoder residual --data_depth 6
python train.py --dataset div2k --encoder residual --data_depth 7
