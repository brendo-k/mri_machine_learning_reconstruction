#python train.py \
#    --data_dir ~/Documents/Coding/data/M4Raw_first_repetition/ \
#    --dataset m4raw \
#    --chans 10 \
#    --cascades 4 \
#    --num_workers 2 \
#    --batch_size 2 \
#    --limit_volumes 0.2 \
#    --contrasts t1 \
#    --run_name "Line Constrained"\
#    --k_loss l1 \
#    --learn_sampling \
#    --sampling_method 1d \
#    --image_scaling_lam_full 1.0e-4 \
#    --image_scaling_full_inv 1.0e-5 \
#    --pass_inverse_data \
#    --line_constrained \
#    --image_loss l1_grad \
#    --all_data_no_grad \
#    --pass_all_data \
#    --image_scaling_lam_inv 1.0e-4 \
#    --seperate_model \
#    --max_epochs 100


export LOSS_OVER_FULL_SET=1
python train.py \
    --data_dir ~/Documents/Coding/data/M4Raw_first_repetition/ \
    --dataset m4raw \
    --chans 10 \
    --cascades 4 \
    --num_workers 2 \
    --batch_size 2 \
    --limit_volumes 0.2 \
    --contrasts t1 \
    --run_name "Loss over full set"\
    --k_loss l1 \
    --learn_sampling \
    --sampling_method 1d \
    --image_scaling_lam_full 1.0e-4 \
    --image_scaling_full_inv 1.0e-5 \
    --pass_inverse_data \
    --line_constrained \
    --image_loss l1_grad \
    --all_data_no_grad \
    --pass_all_data \
    --image_scaling_lam_inv 1.0e-4 \
    --seperate_model \
    --max_epochs 50

