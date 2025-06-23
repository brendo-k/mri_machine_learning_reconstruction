SIG_SLOPE1=(5 10 50 100)
SIG_SLOPE2=(200 300 400 1000)
for SIG1 in "${SIG_SLOPE1[@]}"; do
    for SIG2 in "${SIG_SLOPE2[@]}"; do
            echo $SIG1 $SIG2
            python /home/brenden/Documents/Coding/python/ml/train.py \
                -c=../configs/learned_partitioning_single_contrast.yaml \
                --chans 10 \
                --cascades 4 \
                --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
                --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
                --R 8 \
                --sampling_method 1d \
                --max_epochs 50 \
                --run_name "lam_full: $LAM_FULL, lam_inv: $LAM_INV, Lam: $LAM" \
                --dataset m4raw \
                --limit_volumes 0.2 \
                --project "test losses" \
                --batch_size 2 \
                --k_loss l1 \
                --image_scaling_lam_full 1e-4 \
                --image_scaling_full_inv 1e-4 \
                --image_scaling_lam_inv 5e-4 \
                --lambda_scaling 0.1 \
                --image_loss l1_grad \
                --pass_all_lines \
                --sigmoid_slope1 $SIG1 \
                --sigmoid_slope2 $SIG2 \
                --warm_start \
                --seperate_model \
                --image_loss_grad_scaling 1 \
                --checkpoint_dir /home/brenden/Documents/Coding/python/ml/checkpoints/
        done
    done
