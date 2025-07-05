#python train.py --data_dir ~/Documents/Coding/data/M4Raw_first_repetition/ --dataset m4raw --chans 10 --cascades 4 --num_workers 2 --batch_size 2 --limit_volumes 0.2 --contrasts t1 --run_name Supervised --supervised --k_loss l1 
#python train.py --data_dir ~/Documents/Coding/data/M4Raw_first_repetition/ --dataset m4raw --chans 10 --cascades 4 --num_workers 2 --batch_size 2 --limit_volumes 0.2 --contrasts t1 --run_name Self-Supervised --k_loss l1 



export REDUCE_LOSS_BY_MASK=0
python train.py \
    --data_dir ~/Documents/Coding/data/M4Raw_first_repetition/ \
    --dataset m4raw \
    --chans 10 \
    --cascades 4 \
    --num_workers 2 \
    --batch_size 2 \
    --limit_volumes 0.2 \
    --contrasts t1 \
    --run_name "Learned Partitioing all data no grad"\
    --k_loss l1 \
    --learn_sampling \
    --image_scaling_lam_full 1.0e-3 \
    --image_scaling_full_inv 8.0e-4 \
    --pass_inverse_data \
    --all_data_no_grad \
    --pass_all_data \
    --image_scaling_lam_inv 1.0e-4 \
    --seperate_model \

