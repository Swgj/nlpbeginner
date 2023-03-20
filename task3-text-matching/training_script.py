import json
import argparse
from Train import run_train

if __name__ == '__main__':
    with open("./train.json", "r") as f:
        train_config = json.load(f)

    train_args = argparse.Namespace(**train_config)

    run_train(train_file=train_args.train_data,
              valid_file=train_args.valid_data,
              embeddings_file=train_args.embeddings,
              target_dir=train_args.target_dir,
              hidden_size=train_args.hidden_size,
              dropout=train_args.dropout,
              num_classes=train_args.num_classes,
              epochs=train_args.epochs,
              batch_size=train_args.batch_size,
              lr=train_args.lr,
              patience=train_args.patience,
              max_grad_norm=train_args.max_gradient_norm,
              checkpoint="./checkpoints/esim_14.pth.tar"
              )