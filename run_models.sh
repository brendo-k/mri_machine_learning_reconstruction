export WARMUP_EPOCHS=20
python train.py \
    -c=configs/learned_partitioning_single_contrast.yaml \
    --chans 18 \
    --cascades 6 \
    --data_dir /home/brenden/Documents/Coding/data/reduced_fastmri/ \
    --test_dir /home/brenden/Documents/Coding/data/reduced_fastmri/ \
    --R 6 \
    --contrast t1 \
    --lr 1e-3 \
    --sampling_method 2d \
    --max_epochs 50 \
    --run_name "Current" \
    --dataset fastmri \
    --limit_volumes 0.2 \
    --batch_size 1 \
    --image_scaling_lam_full 1.0e-3 \
    --image_scaling_full_inv 1.0e-3 \
    --image_scaling_lam_inv 1.0e-3 \
    --image_loss l1_grad \
    --warm_start \
    --warmup_training \
    --lambda_scaling 0.9


