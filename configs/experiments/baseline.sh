# mc methods
python train.py -c configs/brats/train_self_supervised_mc.yaml --sampling_method 2d --run_name 2d_mc --project baselines
python train.py -c configs/brats/train_self_supervised_mc.yaml --sampling_method 1d --run_name 1d_mc --project baselines

# sc methods
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 2d --run_name 2d_sc_t1 --contrasts t1 --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 2d --run_name 2d_sc_t2 --contrasts t2 --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 2d --run_name 2d_sc_t1ce --contrasts t1ce --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 2d --run_name 2d_sc_flair --contrasts flair --project baselines

python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 1d --run_name 1d_sc_t1 --contrasts t1 --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 1d --run_name 1d_sc_t2 --contrasts t2 --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 1d --run_name 1d_sc_t1ce --contrasts t1ce --project baselines
python train.py -c configs/brats/train_self_supervised_sc.yaml --sampling_method 1d --run_name 1d_sc_flair --contrasts flair --project baselines