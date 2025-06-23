IMAGE_LOSS_LAM_FULL=(5e-4)
IMAGE_LOSS_LAM_INV=(1e-5 5e-5 1e-4)
LAMBDA_SCALING=(0.1 0.25 0.5 1)

for LAM_FULL in "${IMAGE_LOSS_LAM_INV[@]}"; do
    for LAM_INV in "${IMAGE_LOSS_LAM_INV[@]}"; do
        for LAM in "${LAMBDA_SCALING[@]}"; do
            echo $LAM_FULL $LAM_INV $LAM
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
                --image_scaling_lam_full $LAM_FULL \
                --image_scaling_full_inv $LAM_FULL \
                --image_scaling_lam_inv $LAM_INV \
                --lambda_scaling $LAM \
                --image_loss l1_grad \
                --pass_all_lines \
                --warm_start \
                --seperate_model \
                --image_loss_grad_scaling 1 \
                --checkpoint_dir /home/brenden/Documents/Coding/python/ml/checkpoints/
            done
        done
    done
