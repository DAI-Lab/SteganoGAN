export CUDA_VISIBLE_DEVICES=1

python train.py --dataset div2k --encoder dense --data_depth 1
python train.py --dataset div2k --encoder dense --data_depth 2
python train.py --dataset div2k --encoder dense --data_depth 3
python train.py --dataset div2k --encoder dense --data_depth 4
python train.py --dataset div2k --encoder dense --data_depth 5
python train.py --dataset div2k --encoder dense --data_depth 6
python train.py --dataset div2k --encoder dense --data_depth 7
