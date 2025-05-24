SAMPLING_METHOD=(1d 2d pi)

cd ../

for SAMP in "${SAMPLING_METHOD[@]}";
do
python train.py \
    -c=configs/ssl_single_contrast.yaml \
    --chans 10 \
    --cascades 4 \
    --data_dir /home/brenden/Documents/Coding/data/M4Raw_first_repetition/ \
    --test_dir /home/brenden/Documents/Coding/data/M4Raw_Averaged/ \
    --R 8 \
    --sampling_method $SAMP \
    --max_epochs 50 \
    --run_name "Baseline $SAMP" \
    --dataset m4raw \
    --limit_volumes 0.2 \
    --project "test double model" \
    --batch_size 2 \
    --k_loss l1 \
    --contrast t1

done
