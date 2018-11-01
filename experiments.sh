python train.py --data_depth 1 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 100.0 --critic_weight 50.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 100.0 --critic_weight 10.0 --dataset div2k

python train.py --data_depth 1 --mse_weight 50.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 50.0 --critic_weight 50.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 50.0 --critic_weight 10.0 --dataset div2k

python train.py --data_depth 1 --mse_weight 10.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 10.0 --critic_weight 50.0 --dataset div2k
python train.py --data_depth 1 --mse_weight 10.0 --critic_weight 10.0 --dataset div2k

# data depth
python train.py --data_depth 2 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 4 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 8 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 16 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 32 --mse_weight 100.0 --critic_weight 100.0 --dataset div2k

# exploration loss
python train.py --data_depth 1 --mse_weight -10.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 1 --mse_weight -50.0 --critic_weight 100.0 --dataset div2k
python train.py --data_depth 1 --mse_weight -100.0 --critic_weight 100.0 --dataset div2k
