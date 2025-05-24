SAMPLING_METHOD=(pi)
SAMP=pi
LAMBDA_SCALING=(0.1 0.5 1)
IMAGE_SCALING_LAM_INV=(1e-3 1e-4 1e-5)
IMAGE_SCALING_FULL_INV=(1e-3 1e-4 1e-5)
IMAGE_SCALING_FULL_LAM=(1e-3 1e-4 1e-5)

cd ../
for LAM_FULL in "${IMAGE_SCALING_FULL_LAM[@]}"; do
    for LAM in "${LAMBDA_SCALING[@]}"; do
        for LAM_INV in "${IMAGE_SCALING_LAM_INV[@]}"; do
            for FULL_INV in "${IMAGE_SCALING_FULL_INV[@]}"; do
                python train.py \
                    -c=configs/learned_partitioning_single_contrast.yaml \
                    --chans 10 \
                    --cascades 4 \
                    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
                    --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
                    --R 8 \
                    --sampling_method $SAMP \
                    --max_epochs 100 \
                    --run_name "Double Model l1l2 $LAM_FULL $LAM $LAM_INV $FULL_INV" \
                    --dataset m4raw \
                    --limit_volumes 0.2 \
                    --project "test double model" \
                    --batch_size 2 \
                    --k_loss l1l2 \
                    --image_scaling_lam_inv $LAM_INV \
                    --image_scaling_lam_full $LAM_FULL \
                    --image_scaling_full_inv $FULL_INV \
                    --image_loss l1_grad \
                    --lambda_scaling $LAM \
                    --seperate_model


                python train.py \
                    -c=configs/learned_partitioning_single_contrast.yaml \
                    --chans 10 \
                    --cascades 4 \
                    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
                    --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
                    --R 8 \
                    --sampling_method $SAMP \
                    --max_epochs 100 \
                    --run_name "Single Model l1l2 $LAM_FULL $LAM $LAM_INV $FULL_INV" \
                    --dataset m4raw \
                    --limit_volumes 0.2 \
                    --project "test double model" \
                    --batch_size 2 \
                    --k_loss l1l2 \
                    --image_scaling_lam_inv $LAM_INV \
                    --image_scaling_lam_full $LAM_FULL \
                    --image_scaling_full_inv $FULL_INV \
                    --lambda_scaling $LAM \
                    --image_loss l1_grad
                done
            done
        done
    done
