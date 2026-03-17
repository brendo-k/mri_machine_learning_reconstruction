#SAMPLING_METHOD=(1d 2d pi)
SAMPLING_METHOD=(pi)

cd ../

for SAMP in "${SAMPLING_METHOD[@]}";
do
python train.py \
    -c=configs/ssl_single_contrast.yaml \
    --chans 18 \
    --cascades  \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --R 6 \
    --sampling_method $SAMP \
    --max_epochs 50 \
    --run_name "Baseline $SAMP" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project "march_17" \
    --batch_size 2 \
    --k_loss l1 \
    --contrast t1 t2
done
