export FINAL_DC_STEP_INFER=1; export FINAL_DC_STEP=1;
python /home/brenden/Documents/Coding/python/ml/train.py \
    -c=configs/learned_partitioning_single_contrast.yaml \
    --chans  10 \
    --cascades  4 \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --R 8 \
    --sampling_method 2d \
    --max_epochs 100 \
    --run_name "LP warm_start seperate_model not_learn_R $FINAL_DC_STEP_INFER final_dc_step $FINAL_DC_STEP" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project  "test double model" \
    --batch_size 2 \
    --k_loss l1 \
    --image_scaling_lam_full 1e-3 \
    --image_scaling_full_inv 1e-3 \
    --image_scaling_lam_inv 1e-4 \
    --lambda_scaling 0.75 \
    --image_loss l1_grad \
    --image_loss_grad_scaling 1 \
    --pass_inverse_data \
    --seperate_model \
    --warm_start

export FINAL_DC_STEP_INFER=1; export FINAL_DC_STEP=1;
python /home/brenden/Documents/Coding/python/ml/train.py \
    -c=configs/learned_partitioning_single_contrast.yaml \
    --chans  10 \
    --cascades  4 \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --R 8 \
    --sampling_method 2d \
    --max_epochs 100 \
    --run_name "LP warm_start not_learn_R $FINAL_DC_STEP_INFER final_dc_step $FINAL_DC_STEP" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project  "test double model" \
    --batch_size 2 \
    --k_loss l1 \
    --image_scaling_lam_full 1e-3 \
    --image_scaling_full_inv 1e-3 \
    --image_scaling_lam_inv 1e-4 \
    --lambda_scaling 0.75 \
    --image_loss l1_grad \
    --image_loss_grad_scaling 1 \
    --pass_inverse_data \
    --warm_start
