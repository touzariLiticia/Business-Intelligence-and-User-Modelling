from comet_ml import Experiment
import numpy as np
import torch.utils.data as tud
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pl_trainer import WhichGameTrainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append('../')
from args import whichgame_args
sys.path.append('../../data_utils')
from dataset import DatasetGetter
from data_utils import populate_classes_lists, MultiDatasetSampler


if __name__ == "__main__":

    args = whichgame_args().parse_args()

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY", "rEy6sPPJyZ6EATCASeGFbeNB8"),
        project_name=args.project_name,
        experiment_name=args.experiment_name,
    )

    args.ckpts_out_dir = os.path.join(
        args.ckpt_dir, args.project_name, args.experiment_name
    )

    (
        ingame_classes,
        outgame_classes,
        ingame_class_to_idx,
        outgame_class_to_idx,
    ) = populate_classes_lists(
        args.data_dir, args.ingame_subdir, args.outgame_cats_subdir
    )

    # args.num_workers = 2
    args.num_workers //= args.n_gpus

    ckpt_to_load_path = None
    # ckpt_to_load_path = "/bigdata0/checkpoints/whichgame-tests/mobilenet-robin-labels/epoch-epoch=00020.ckpt"

    dataset_getter = DatasetGetter(
        args.data_dir,
        ingame_class_to_idx,
        args.labels_dir,
        args.ingame_subdir,
        args.outgame_subdir,
        args.outgame_cats_subdir,
        valid_test_twids_paths=(args.valid_twids_path, args.test_twids_path),
        outgame_class_to_idx=outgame_class_to_idx,
    )
    train_dataset, valid_datasets = dataset_getter()

    model = WhichGameTrainer(
        backbone_name=args.backbone_name,
        num_classes=len(ingame_classes),
        num_outgame_subclasses=len(outgame_classes),
        dropout_fts=args.dropout_fts,
        pretrained=args.pretrained,
        lr=args.lr,
        epochs=args.epochs,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpts_out_dir, filename="epoch-{epoch:05d}", save_top_k=-1
    )
    trainer = Trainer(
        gpus=args.n_gpus,
        max_epochs=args.epochs,
        accelerator="ddp",
        logger=comet_logger,
        resume_from_checkpoint=ckpt_to_load_path,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        replace_sampler_ddp=False,
        # limit_train_batches=10,
        # limit_val_batches=10,
    )
    train_loader = tud.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=MultiDatasetSampler._worker_init_fn,
    )
    valid_loaders = [
        tud.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for ds in valid_datasets
    ]
    trainer.fit(model, train_loader, valid_loaders)
