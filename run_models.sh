sampling=(1d 2d pi)


for sampling in ${sampling[@]}; do
    #echo $grad
    python train.py \
        -c=configs/learned_partitioning_single_contrast.yaml \
        --chans 10 \
        --cascades 4 \
        --data_dir /home/brenden/Documents/Coding/data/sim_subset_1e-2_4chan/ \
        --test_dir /home/brenden/Documents/Coding/data/sim_subset_1e-2_4chan/ \
        --R 6 \
        --contrast t1 \
        --lr 1e-3 \
        --sampling_method $sampling \
        --max_epochs 50 \
        --run_name $sampling 
    done
