# training script for resnet50, learned deferral one-stage and resnet50 dropout
from deferral_model import Model
from McDropout import McDropoutModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
# from data.AIROGS_dataloader import AIROGS
from torchvision import datasets, transforms
import torch

def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store",
        default=0,
        type=int,
        help="Random seed for pl.seed_everything function.",
    )
    
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        action="store",
        default=4,
        type=int,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--type",
        dest="type",
        action="store",
        default="default",
        type=str,
        help="Type of model. 'defer' or 'default' or 'dropout'",
    )

    parser.add_argument(
        "--c",
        dest="c",
        action="store",
        default=0.2,
        type=float,
        help="Deferral cost",
    )

    args = parser.parse_args()

    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [
            git_hash,
            f"seed={args.random_seed}",
            args.type,
            str(args.c),
            human_readable_extra,
            f"bs={args.batch_size}",
        ]
    )

    pl.seed_everything(seed=args.random_seed)

    # path = "/AIROGS.h5"
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    valid_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_train, download=True)
    # train_dataset = AIROGS(file_path=path, t="train", transform=None)
    # valid_dataset = AIROGS(file_path=path, t="val", transform=None)

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    def get_oversampler(dataset):
        # oversample minority class (RG)
        class_counts = torch.bincount(torch.tensor(dataset.labels))  
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[dataset.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return sampler

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=False#, sampler=get_oversampler(train_dataset)
        )
    valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, drop_last=False#, sampler=get_oversampler(valid_dataset)
    )

    logger = TensorBoardLogger(
        save_dir="./runs", name=experiment_name, default_hp_metric=False
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            filename="best-loss-{epoch}-{step}",
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val/pauc",
            filename="best-pauc-{epoch}-{step}",
            mode="max",
        )
    ]
    if args.type == "default":
        model = Model(3,2,"default", args.c)
    elif args.type == "defer":
        model = Model(3,3, "defer", args.c)
    elif args.type == "dropout":
        model = McDropoutModel(3, 2)
    else:
        print("Wrong model type!")


    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        callbacks=checkpoint_callbacks,
        min_epochs=10,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )