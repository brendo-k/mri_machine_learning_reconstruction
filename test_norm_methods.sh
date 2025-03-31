python train.py -c configs/brats/train_ssl_mc.yaml --norm_method k --run_name k
python train.py -c configs/brats/train_ssl_mc.yaml --norm_method image_mean --run_name image_mean
python train.py -c configs/brats/train_ssl_mc.yaml --norm_method img --run_name img
python train.py -c configs/brats/train_ssl_mc.yaml --norm_method std --run_name std
