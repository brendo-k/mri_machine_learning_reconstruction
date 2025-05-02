grad_scaling_delta=(7.0e-4 4.0e-4 1.0e-4)

for grad_delta in ${grad_scaling_delta[@]}; do
    #echo $grad
    python train.py \
        -c=configs/learned_partitioning_single_contrast.yaml \
        --chans 10 \
        --cascades 4 \
        --data_dir /home/brenden/Documents/Coding/data/sim_subset_1e-2_4chan/ \
        --test_dir /home/brenden/Documents/Coding/data/sim_subset_1e-2_4chan/ \
        --R 6 \
        --contrast t1 \
        --warm_start \
        --lr 1e-3 \
        --sampling_method 1d \
        --max_epochs 50 \
        --image_loss_grad_scaling 0.5 \
        --pass_all_lines \
        --ssim_scaling_set 4.0e-4 \
        --ssim_scaling_full 1.0e-3 \
        --ssim_scaling_inverse 4.0e-4 \
        --ssim_scaling_delta $grad_delta
        --run_name "delta $grad_delta"

    done
