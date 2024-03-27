import pytorch_lightning as pl

def test_pass_through_test():
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=[tb_logger, csv_logger, wandb_logger], limit_train_batches=args.limit_train_batches)
