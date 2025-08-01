values1=(5.0e-3)
values2=(5.0e-4 1.0e-4)

for var1 in ${values1[@]}; do 
    python train.py \
        -c=./configs/learned_partitioning_single_contrast.yaml \
        --image_scaling_lam_inv $var1 \
        --image_scaling_full_inv $var1 \
        --image_scaling_lam_full $var1 \
        --run_name "lr 1e-3" \
        --lr 1e-3 \
        --warmup_adam

    python train.py \
        -c=./configs/learned_partitioning_single_contrast.yaml \
        --image_scaling_lam_inv $var1 \
        --image_scaling_full_inv $var1 \
        --image_scaling_lam_full $var1 \
        --run_name "lr 5e-4" \
        --lr 5e-4 \
        --warmup_adam

    python train.py \
        -c=./configs/learned_partitioning_single_contrast.yaml \
        --image_scaling_lam_inv $var1 \
        --image_scaling_full_inv $var1 \
        --image_scaling_lam_full $var1 \
        --run_name "lr 5e-3" \
        --lr 5e-3 \
        --warmup_adam
done
