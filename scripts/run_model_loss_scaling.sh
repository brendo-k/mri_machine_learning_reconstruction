image_loss_scaling=(1e-2 1e-3 1e-4)
grad_scaling=(0 0.5 1 2)

for GRAD in ${grad_scaling[@]}; do
    for IMAGE_LOSS in ${image_loss_scaling[@]}; do

        python /home/brenden/Documents/Coding/python/ml/train.py \
            -c=../configs/learned_partitioning_single_contrast.yaml \
            --chans 10 \
            --cascades 4 \
            --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
            --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
            --R 8 \
            --sampling_method 1d \
            --max_epochs 100 \
            --run_name "Double grad $GRAD image $IMAGE_LOSS" \
            --dataset m4raw \
            --limit_volumes 0.2 \
            --project "test losses" \
            --batch_size 2 \
            --k_loss l1 \
            --image_scaling_lam_full $IMAGE_LOSS \
            --image_scaling_full_inv $IMAGE_LOSS \
            --image_scaling_lam_inv 0 \
            --lambda_scaling 1 \
            --image_loss l1_grad \
            --pass_all_lines \
            --warm_start \
            --seperate_model \
            --all_data_no_grad \
            --image_loss_grad_scaling $GRAD \
            --checkpoint_dir /home/brenden/Documents/Coding/python/ml/checkpoints/
        done
    done
