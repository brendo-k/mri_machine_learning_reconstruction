for conv in true false; do
    if [ "$conv" = true ]; then
        FLAG="--conv_after_upsample"
    else
        FLAG=""
    fi
    for method in "max" "bilinear" "conv" "conv_init"; do
        python train.py \
        -c configs/brats/train_ssl_mc.yaml \
        --upsample_method $method \
        --max_epochs 50 \
        --run_name $method \
        --limit_volumes 0.25 \
        $FLAG
    done
done