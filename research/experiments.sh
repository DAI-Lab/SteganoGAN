source activate pytorch_p36
echo $CUDA_VISIBLE_DEVICES
python train.py --dataset mscoco --encoder dense --data_depth 1 --epochs 1
python train.py --dataset mscoco --encoder dense --data_depth 2 --epochs 1
python train.py --dataset mscoco --encoder dense --data_depth 3 --epochs 1
python train.py --dataset mscoco --encoder dense --data_depth 4 --epochs 1
python train.py --dataset mscoco --encoder dense --data_depth 5 --epochs 1
python train.py --dataset mscoco --encoder dense --data_depth 6 --epochs 1
