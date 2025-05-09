export WARMUP_EPOCHS=10

NORM_METHODS=(k img image_mean norm_l2 std)
SAMP_METHOD=(2d 1d pi)

SAMP=1d

export REDUCE_LOSS_BY_MASK=true
python train.py \
    -c=configs/ssl_single_contrast.yaml \
    --chans 18 \
    --cascades 6 \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
    --R 8 \
    --contrast t1 t2 flair t1ce\
    --lr 1e-3 \
    --sampling_method $SAMP \
    --max_epochs 50 \
    --run_name "norm by mask" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project norm_masks \
    --batch_size 5

export REDUCE_LOSS_BY_MASK=false
python train.py \
    -c=configs/ssl_single_contrast.yaml \
    --chans 18 \
    --cascades 6 \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
    --R 8 \
    --contrast t1 t2 flair t1ce\
    --lr 1e-3 \
    --sampling_method $SAMP \
    --max_epochs 50 \
    --run_name "base" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project norm_masks \
    --batch_size 5 \
