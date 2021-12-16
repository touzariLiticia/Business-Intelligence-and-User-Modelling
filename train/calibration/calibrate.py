import comet_ml 
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from glob import glob
from onnx import onnx_pb
from coremltools.converters.onnx import convert
from sklearn.metrics import roc_auc_score
from pytorch_lightning import Trainer
import torch
from torch.nn import functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
from tqdm import tqdm
import json
import argparse
import pylab as plt
import numpy as np

import sys
import os
path = os.path.abspath(os.path.realpath(__file__)+"/../..")
sys.path.append(path)
from train.calibration.calibrate_metrics import *
from train.args import whichgame_args
from data_util.dataset import DatasetGetter
from data_util.data_utils import populate_classes_lists
from train.net.network import Net

def get_commit_hash():
    with open("../commithash", "r") as f:
        commithash = f.read()
        if commithash.endswith("\n"):
            commithash = commithash[:-1]
    return commithash


def inference_dataset(
    dataset, model, args, every=1, ret_xs=False, ret_just_one_x=False
):
    if every > 1:
        dataset = tud.Subset(0, len(dataset), every)
    kwargs = dict(
        pin_memory=args.device != "cpu", num_workers=args.num_workers
    )
    loader = tud.DataLoader(dataset, batch_size=args.batch_size, **kwargs)

    if ret_just_one_x:
        ret_xs = True
    model.eval()
    (
        xs,
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    ) = [[] for _ in range(8)]
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["images"]
            y_ingame = batch["y_ingame"]
            y_whichgame = batch["y_whichgame"]
            y_subclass_if_outgame = batch["y_subclass_if_outgame"]
            (
                x_ingame,
                x_game,
                x_outgame_subclasses,
                ftrs_for_distill,
                x_ftrs,
            ) = model(x.to(args.device))
            r = x_ftrs
            p_ingame = x_ingame.softmax(1)
            p_whichgame = x_game.softmax(1)
            p_subclass_if_outgame = x_outgame_subclasses.softmax(1)
            if ret_xs:
                if not ret_just_one_x or not len(xs):
                    xs.append(x.cpu())
            fts.append(r)
            ys_ingame.append(y_ingame)
            ys_whichgame.append(y_whichgame)
            ys_subclass_if_outgame.append(y_subclass_if_outgame)
            ps_ingame.append(p_ingame)
            ps_whichgame.append(p_whichgame)
            ps_subclass_if_outgame.append(p_subclass_if_outgame)
    (
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    ) = [
        torch.cat(_, 0)
        for _ in [
            ys_ingame,
            ys_whichgame,
            ys_subclass_if_outgame,
            ps_ingame,
            ps_whichgame,
            ps_subclass_if_outgame,
            fts,
        ]
    ]
    if ret_xs:
        xs = torch.cat(xs, 0)
        return (
            xs,
            ys_ingame,
            ys_whichgame,
            ys_subclass_if_outgame,
            ps_ingame,
            ps_whichgame,
            ps_subclass_if_outgame,
            fts,
        )
    return (
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    )


class CalibrateFC(LightningModule):
    def __init__(
        self,
        X,
        Y,
        X_valid,
        Y_valid,
        dim_out,
        dim_in=None,
        lr=0.1,
        momentum=0.9,
        batch_size=1024,
        weight_decay=1e-3,
        init_model=None,
        n_bins=10
    ):
        super().__init__()
        self.X = X
        self.Y = Y
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.lr = lr
        self.momentum = momentum
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in is None:
            self.dim_in = self.X.shape[1]
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.fc = torch.nn.Linear(self.dim_in, self.dim_out)
        if init_model is not None:
            self.fc.weight.data *= 0
            self.fc.weight.data += init_model.weight.data.to(
                self.fc.weight.device
            )
            self.fc.bias.data *= 0
            self.fc.bias.data += init_model.bias.data.to(self.fc.weight.device)
        self.ece = ECELoss(n_bins=n_bins)
        self.ada_ece = AdaptiveECELoss(n_bins=n_bins)
        self.classwise_ece = ClasswiseECELoss(dim_out, n_bins=n_bins)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        #self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        dataset = tud.TensorDataset(self.X, self.Y)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=0, shuffle=True
        )
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, reduction="none")
        val_acc = y_hat.argmax(1).eq(y)
        ece_loss = self.ece(y_hat, y)
        ada_ece_loss = self.ada_ece(y_hat, y)
        classwise_ece_loss = self.classwise_ece(y_hat, y)
        return {"val_loss": val_loss, "val_acc": val_acc, "ece_loss": ece_loss, "ada_ece_loss": ada_ece_loss, "classwise_ece_loss":classwise_ece_loss}


    def validation_epoch_end(self, outputs):

        avg_loss = torch.cat([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.cat([x["val_acc"] for x in outputs]).float().mean()
        avg_ece = torch.cat([x["ece_loss"] for x in outputs]).float().mean()
        avg_ada_ece = torch.cat([x["ada_ece_loss"] for x in outputs]).float().mean()
        avg_classwise_ece = torch.cat([x["classwise_ece_loss"] for x in outputs]).float().mean()
        logs = {"val_loss": avg_loss, "val_acc": avg_acc, "ece_loss": avg_ece, "ada_ece_loss": avg_ada_ece, "classwise_ece_loss":avg_classwise_ece}
        #[self.log(k, v) for k, v in logs.items()]
        for k, v in logs.items():
            self.log(
                "validation " + k,
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        print(logs)
        return {"val_loss": avg_loss}

    def val_dataloader(self):
        dataset = tud.TensorDataset(self.X_valid, self.Y_valid)
        loader = DataLoader(
            dataset, batch_size=len(dataset), num_workers=0, drop_last=False
        )
        return loader


class CalibrateFCWhichGame(CalibrateFC):
    def __init__(
        self,
        model_ig,
        Y_valid_ig,
        X,
        Y,
        X_valid,
        Y_valid,
        dim_out,
        dim_in=None,
        lr=0.1,
        momentum=0.9,
        batch_size=1024,
        weight_decay=1e-3,
        init_model=None,
        weighted_training=True,
        n_bins=10
    ):
        super().__init__(
            X,
            Y,
            X_valid,
            Y_valid,
            dim_out,
            dim_in,
            lr,
            momentum,
            batch_size,
            weight_decay,
            init_model,
            n_bins,
        )
        self.model_ig = model_ig
        self.Y_valid_ig = Y_valid_ig
        self.weighted_training = weighted_training

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        with torch.no_grad():
            p_ingame = torch.softmax(self.model_ig(x), dim=1)[:, 1]
            weights = p_ingame / p_ingame.sum()
            if not self.weighted_training:
                weights = weights * 0 + 1
        loss = (F.cross_entropy(y_hat, y, reduction="none") * weights).sum()
        #self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, y_ig = batch
        y_hat = self(x)
        ig = y_ig == 1
        val_loss = F.cross_entropy(y_hat[ig], y[ig], reduction="none")
        val_acc = y_hat[ig].argmax(1).eq(y[ig])
        ece_loss = self.ece(y_hat, y)
        ada_ece_loss = self.ada_ece(y_hat, y)
        classwise_ece_loss = self.classwise_ece(y_hat, y)
        return {"val_loss": val_loss, "val_acc": val_acc, "ece_loss": ece_loss, "ada_ece_loss": ada_ece_loss, "classwise_ece_loss":classwise_ece_loss}

    def val_dataloader(self):
        dataset = tud.TensorDataset(
            self.X_valid, self.Y_valid, self.Y_valid_ig
        )
        loader = DataLoader(
            dataset, batch_size=len(dataset), num_workers=0, drop_last=False
        )
        return loader

class CalibrateWithTemperature(CalibrateFC):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    """def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)"""

    def __init__(
        self,
        X,
        Y,
        X_valid,
        Y_valid,
        dim_out,
        dim_in=None,
        lr=0.1,
        momentum=0.9,
        batch_size=1024,
        weight_decay=1e-3,
        init_model=None,
        n_bins=10
    ):
        super().__init__(
            X,
            Y,
            X_valid,
            Y_valid,
            dim_out,
            dim_in,
            lr,
            momentum,
            batch_size,
            weight_decay,
            init_model,
            n_bins,
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.fc(x)
        return self.temperature_scale(logits)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


    def configure_optimizers(self):
        #return torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=50)
        
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

class CalibrateWithFocallLoss(CalibrateWithTemperature):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    """def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)"""

    def __init__(
        self,
        X,
        Y,
        X_valid,
        Y_valid,
        dim_out,
        dim_in=None,
        lr=0.1,
        momentum=0.9,
        batch_size=1024,
        weight_decay=1e-3,
        init_model=None,
        n_bins=10,
        gamma= 0
    ):
        super().__init__(
            X,
            Y,
            X_valid,
            Y_valid,
            dim_out,
            dim_in,
            lr,
            momentum,
            batch_size,
            weight_decay,
            init_model,
            n_bins,
        )
        self.gamma = gamma
        self.loss = FocalLoss(gamma=gamma)
        
    def forward(self, x):
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        #self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        val_acc = y_hat.argmax(1).eq(y)
        ece_loss = self.ece(y_hat, y)
        ada_ece_loss = self.ada_ece(y_hat, y)
        classwise_ece_loss = self.classwise_ece(y_hat, y)
        return {"val_loss": val_loss, "val_acc": val_acc, "ece_loss": ece_loss, "ada_ece_loss": ada_ece_loss, "classwise_ece_loss":classwise_ece_loss}


    def configure_optimizers(self):
        
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )    

def set_temperature(model, valid_loader):
        
        # Tune the tempearature of the model (using the validation set).
        # We're going to set it to optimize NLL.
        # valid_loader (DataLoader): validation set loader
       
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(model.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

class SoftmaxOutputPClass1(torch.nn.Module):
    def __init__(self):
        super(SoftmaxOutputPClass1, self).__init__()

    def forward(self, x):
        return torch.softmax(x, 1)


class WrapIt255(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        inv = torch.FloatTensor([1 / 0.229, 1 / 0.224, 1 / 0.225]) / 255.0
        mean = torch.FloatTensor([0.485, 0.456, 0.406]) * 255.0
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv.weight.data = torch.diag(inv).view(3, 3, 1, 1)
        self.conv.bias.data = -mean * inv
        self.m = m
        self.sm = SoftmaxOutputPClass1()

    def forward(self, x):
        x = self.conv(x)
        ig, which, og_subcats = self.m(x)[:3]
        return self.sm(ig), self.sm(which), self.sm(og_subcats)


class WrapIt(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.sm = SoftmaxOutputPClass1()

    def forward(self, x):
        ig, which, og_subcats = self.m(x)[:3]
        return self.sm(ig), self.sm(which), self.sm(og_subcats)


if __name__ == "__main__":
    
    # Load args and classes
    args = whichgame_args().parse_args()
    args.device = "cuda" if args.n_gpus > 0 else "cpu"
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
    # Load up the latest checkpoint from the checkpoints dir
    list_of_files = glob(os.path.join(args.ckpts_out_dir, "*.ckpt"))
    assert len(list_of_files), f"No checkpoints found in {args.ckpts_out_dir}"

    ckpt_path = os.path.join(args.ckpts_out_dir, 'epoch-epoch=00049.ckpt')
    print(f"Will load checkpoint from {ckpt_path}")
    model = Net(
        backbone_name=args.backbone_name,
        num_classes=len(ingame_classes),
        num_outgame_subclasses=len(outgame_classes),
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")["state_dict"]
    )
    model = model.eval().to(args.device)
    print(f"Model state succesfully loaded from {ckpt_path}")

    ########################
    #    DATA
    ########################

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

    unlabeled_datasets = {}
    _ = dataset_getter.get_unlabeled_ingame_dataset("valid")
    # Reduce the size of the unlabeled dataset to make validation epoch faster
    unlabeled_valid_dataset_reduce_factor = 100
    ingame_unlabeled_reduced_dataset = tud.Subset(
        _, np.arange(0, len(_), unlabeled_valid_dataset_reduce_factor)
    )
    print(
        f"len(ingame_unlabeled_reduced_dataset) = {len(ingame_unlabeled_reduced_dataset)}"
    )

    (
        ys_ingame_u,
        ys_whichgame_u,
        ys_subclass_if_outgame_u,
        ps_ingame_u,
        ps_whichgame_u,
        ps_subclass_if_outgame_u,
        fts_u,
    ) = inference_dataset(ingame_unlabeled_reduced_dataset, model, args)


    ds = dataset_getter.get_outgame_dataset_cats("valid")
    ds2 = dataset_getter.get_outgame_dataset_cats(
        "valid", apply_augmentation=True
    )
    ds = tud.ConcatDataset([ds, ds2])
    (
        _,
        _,
        ys_subclass_if_outgame_valid,
        _,
        _,
        ps_subclass_if_outgame_valid,
        fts_outgame_valid,
    ) = inference_dataset(ds, model, args)
    ds = dataset_getter.get_outgame_dataset_cats("test")
    (
        _,
        _,
        ys_subclass_if_outgame_test,
        _,
        _,
        ps_subclass_if_outgame_test,
        fts_outgame_test,
    ) = inference_dataset(ds, model, args)


    ingame_labeled_dataset = tud.ConcatDataset(
        [
            # dataset_getter.get_labeled_ingame_dataset("train"),
            dataset_getter.get_labeled_ingame_dataset("valid"),
        ]
    )

    (
        ys_ingame,
        ys_whichgame,
        _,
        ps_ingame,
        ps_whichgame,
        _,
        fts,
    ) = inference_dataset(ingame_labeled_dataset, model, args)

    test_dataset = dataset_getter.get_labeled_ingame_dataset("test")
    (
        xs,
        ys_ingame_t,
        ys_whichgame_t,
        ys_subclass_if_outgame_t,
        ps_ingame_t,
        ps_whichgame_t,
        ps_subclass_if_outgame_t,
        fts_t,
    ) = inference_dataset(test_dataset, model, args, ret_just_one_x=True)


    ########################
    #    CALIBRATORS
    ########################
    
    calibrator_outgame_subclasses = CalibrateWithFocallLoss(
        fts_outgame_valid,
        ys_subclass_if_outgame_valid,
        fts_outgame_test,
        ys_subclass_if_outgame_test,
        2,
        lr=0.01,
        weight_decay=1e-3,
        init_model=model.fc_outgame_subclasses,
    )
    experiment = CometLogger(
        api_key="rEy6sPPJyZ6EATCASeGFbeNB8",
        project_name="whichgame-tests",
        experiment_name="calibration_outgame_fl_gamma="+str(calibrator_outgame_subclasses.gamma))

    trainer = Trainer(
        max_epochs=15, gpus=1, num_nodes=1, num_sanity_val_steps=-1, logger = experiment
    )
    print("Fitting calibrator_outgame_subclasses...")
    trainer.fit(calibrator_outgame_subclasses)
    
    calibrator_ingame = CalibrateWithFocallLoss(
        fts,
        ys_ingame,
        fts_t,
        ys_ingame_t,
        2,
        lr=0.01,
        weight_decay=1e-3,
        init_model=model.fc_ingame,
    )
    
    experiment = CometLogger(
        api_key="rEy6sPPJyZ6EATCASeGFbeNB8",
        project_name="whichgame-tests",
        experiment_name="calibration_ingame_fl_gamma="+str(calibrator_ingame.gamma)) 
    
    trainer = Trainer(
        max_epochs=15, gpus=1, num_nodes=1, num_sanity_val_steps=-1, logger = experiment
    )
    print("Fitting calibrator_ingame...")
    trainer.fit(calibrator_ingame)
    
    assert len(ingame_classes) == model.fc_game.bias.shape[0]
    assert len(ingame_classes) == len(
        np.unique(torch.cat([ys_whichgame, ys_whichgame_u]))
    )
    assert all(
        np.arange(len(ingame_classes))
        == np.unique(torch.cat([ys_whichgame, ys_whichgame_u]))
    )

    """calibrator_whichgame = CalibrateFCWhichGame(
        calibrator_ingame,
        ys_ingame_t,
        torch.cat([fts]),
        torch.cat([ys_whichgame]),
        fts_t,
        ys_whichgame_t,
        len(ingame_classes),
        lr=0.01,
        init_model=model.fc_game,
        weight_decay=0.0,
    )"""
    
    calibrator_whichgame = CalibrateWithFocallLoss(
        torch.cat([fts]),
        torch.cat([ys_whichgame]),
        fts_t,
        ys_whichgame_t,
        len(ingame_classes),
        lr=0.01,
        init_model=model.fc_game,
        weight_decay=0.0,
    )

    experiment = CometLogger(
        api_key="rEy6sPPJyZ6EATCASeGFbeNB8",
        project_name="whichgame-tests",
        experiment_name="calibration_whichgame_fl_gamma="+str(calibrator_whichgame.gamma))

    trainer = Trainer(max_epochs=15, gpus=1, num_nodes=1, logger = experiment)
    print("Fitting calibrator_whichgame...")
    trainer.fit(calibrator_whichgame)



    model.fc_outgame_subclasses.bias.data *= 0
    model.fc_outgame_subclasses.bias.data += calibrator_outgame_subclasses.fc.bias.data.to(
        model.fc_outgame_subclasses.bias.device
    )
    model.fc_outgame_subclasses.weight.data *= 0
    model.fc_outgame_subclasses.weight.data += calibrator_outgame_subclasses.fc.weight.data.to(
        model.fc_outgame_subclasses.weight.device
    )

    model.fc_game.bias.data *= 0
    model.fc_game.bias.data += calibrator_whichgame.fc.bias.data.to(
        model.fc_game.bias.device
    )
    model.fc_game.weight.data *= 0
    model.fc_game.weight.data += calibrator_whichgame.fc.weight.data.to(
        model.fc_game.weight.device
    )

    model.fc_ingame.bias.data *= 0
    model.fc_ingame.bias.data += calibrator_ingame.fc.bias.data.to(
        model.fc_ingame.bias.device
    )
    model.fc_ingame.weight.data *= 0
    model.fc_ingame.weight.data += calibrator_ingame.fc.weight.data.to(
        model.fc_ingame.weight.device
    )
    model.eval()

    def get_trace_path(
        info_str, run_name=args.experiment_name, commithash=get_commit_hash(),
    ):
        return f"{run_name}_{commithash}{info_str}.pth"

    # Save the classnames
    os.makedirs("calibrate_outs", exist_ok=True)
    with open(
        os.path.join("calibrate_outs", get_trace_path("")[:-4])
        + "_classes.json",
        "w",
    ) as f:
        json.dump([i for i in ingame_classes], f)

    torch.save(
        model.cpu().eval().state_dict(),
        os.path.join("calibrate_outs", get_trace_path("_calibrated")),
    )

    # Save the model trace (imagenet normalized version)
    trace = torch.jit.trace(WrapIt(model.cpu()), torch.randn(xs[:1].shape))
    trace_path = get_trace_path("")
    torch.jit.save(trace, os.path.join("calibrate_outs", trace_path))

    # Save the model trace (RGB255 input version)
    trace = torch.jit.trace(WrapIt255(model.cpu()), torch.randn(xs[:1].shape))
    trace_path = get_trace_path("-rgb255in")
    torch.jit.save(trace, os.path.join("calibrate_outs", trace_path))

    coreml_path = os.path.join(
        "calibrate_outs",
        trace_path.replace("mdl", "coreml").replace(".pth", ".mlmodel"),
    )

    # Now ONNX export for coreml
    onnx_path = coreml_path[:-7] + "onnx"
    model.eval()
    m = WrapIt255(model.cpu())
    input_names = ["image"]
    output_names = ["out_game_in_game_proba", "game_proba", "nude_safe_proba"]
    with torch.no_grad():
        torch_out = torch.onnx.export(
            m,
            xs[:1],
            onnx_path,
            export_params=True,
            opset_version=7,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    with open("conv_dict.json", "r") as f:
        conv_dict = json.load(f)
    coreml_model = convert(
        model=onnx_path,
        image_input_names=input_names,
        minimum_ios_deployment_target="12",
    )
    coreml_model.author = "Liam Schoneveld"
    coreml_model.user_defined_metadata[
        "whichgame-ai commit hash"
    ] = get_commit_hash()
    coreml_model.user_defined_metadata["game_codes"] = ",".join(
        [conv_dict[k] for k in ingame_classes]
    )
    coreml_model.user_defined_metadata["outgame_classes"] = ",".join(
        outgame_classes
    )
    coreml_model.short_description = "TESTING version of whichgame-porn model, if in-game, detects between 79 games, if out-game, detects between porn/safe"
    coreml_model.save(coreml_path,)
    print("Saved coreml model to", coreml_path)

