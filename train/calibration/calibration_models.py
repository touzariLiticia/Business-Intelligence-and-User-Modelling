from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
import pylab as plt
import numpy as np
import torch.utils.data as tud
from torch.utils.data import DataLoader
from calibrate_metrics import *

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
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.fc(x)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


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
        gamma= 2
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

def set_temperature(model, valid_loader, device):
        
        # Tune the tempearature of the model (using the validation set).
        # We're going to set it to optimize NLL.
        # valid_loader (DataLoader): validation set loader
       
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = model(input).to(device)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(model.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(model.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(model.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % model.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return model



