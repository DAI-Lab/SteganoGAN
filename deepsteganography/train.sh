#python train.py --dataset data/caltech256 --data_depth 1 --hidden_dim 32
#python train.py --dataset data/caltech256 --data_depth 1 --hidden_dim 64
#python train.py --dataset data/caltech256 --data_depth 2 --hidden_dim 32
#python train.py --dataset data/caltech256 --data_depth 2 --hidden_dim 64
#python train.py --dataset data/caltech256 --data_depth 4 --hidden_dim 32
#python train.py --dataset data/caltech256 --data_depth 4 --hidden_dim 64
#python train.py --dataset data/caltech256 --data_depth 8 --hidden_dim 32
#python train.py --dataset data/caltech256 --data_depth 8 --hidden_dim 64
python train.py --dataset data/caltech256 --data_depth 16 --hidden_dim 32
python train.py --dataset data/caltech256 --data_depth 16 --hidden_dim 64
python train.py --dataset data/caltech256 --data_depth 32 --hidden_dim 32
python train.py --dataset data/caltech256 --data_depth 32 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 1 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 1 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 2 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 2 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 4 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 4 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 8 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 8 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 16 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 16 --hidden_dim 64
python train.py --dataset data/mscoco --data_depth 32 --hidden_dim 32
python train.py --dataset data/mscoco --data_depth 32 --hidden_dim 64

python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 1 --hidden_dim 32
python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 2 --hidden_dim 32
python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 4 --hidden_dim 32
python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 8 --hidden_dim 32
python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 16 --hidden_dim 32
python train.py --dataset data/caltech256 --test_dataset data/mscoco --data_depth 32 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 1 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 2 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 4 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 8 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 16 --hidden_dim 32
python train.py --dataset data/mscoco --test_dataset data/caltech256 --data_depth 32 --hidden_dim 32
