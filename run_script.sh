python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_delta 0.1 --run_name "0.1"
python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_delta 0 --run_name "0"
python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_delta 1 --run_name "1"
python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_delta 10 --run_name "10"
python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_full 1e-5 --run_name "1e-5"
python train.py -c configs/brats/train_learn_ssl_pass_all_3.yaml --ssim_scaling_full 1e-4 --run_name "1e-4"
