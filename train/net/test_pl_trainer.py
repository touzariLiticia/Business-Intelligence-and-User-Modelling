from comet_ml import Experiment
import numpy as np
import torch.utils.data as tud
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from pl_trainer import WhichGameTrainer


class TestPLModule(WhichGameTrainer):
    def __init__(self):
        super().__init__("mobilenet_v2", 2, 2)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1)[:, :10])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)
        return optimizer

    def calc_losses(self, batch):
        x = batch["images"]
        y = batch["labels"]
        p = self(x)
        losses, to_log = {}, {}
        losses["xent"] = F.cross_entropy(p, y), len(p)
        losses["max_p0"] = (-(p[:, 0] ** 2)).mean(), len(p)
        losses["_average_p"] = p.mean(), len(p)
        conf_ingame = p[:, 1] > 0.5
        losses["_sum_conf_ingame"] = sum(conf_ingame), len(conf_ingame)
        losses["loss"] = losses["xent"][0] + losses["max_p0"][0], len(p)
        losses["loss"] = self._calc_overall_loss_from_losses(losses), len(x)
        return losses


class FakeDataset(tud.Dataset):
    def __init__(self, l):
        self.len = l

    def __getitem__(self, i):
        return {
            "images": torch.randn([3, 10, 10]),
            "labels": torch.randint(0, 5, [1]).item(),
        }

    def __len__(self):
        return self.len


class LessFakeDataset(tud.Dataset):
    def __init__(self, l, mode=0):
        self.len = l
        self.mode = mode

    def __getitem__(self, i):
        l = {}
        l["y_ingame"] = np.random.choice([0, 1, -99999])
        if self.mode == 1:
            l["y_ingame"] = 1
        elif self.mode == 2:
            l["y_ingame"] = 0
        x = torch.rand([3, 200, 200]) * 0.1 - 0.05
        if l["y_ingame"] == 1 or l["y_ingame"] == -99999:
            x[0] += 1
            # If ingame or ingame-unknown we will know whichgame
            l["y_whichgame"] = np.random.choice([0, 1, 2, 3])
            x[1] += l["y_whichgame"] / 2 - 2
            # And we wont have an outgame class
            l["y_subclass_if_outgame"] = -99999
        else:
            x[0] += -1
            # If definitely outgame, we wont have a whichgame class
            l["y_whichgame"] = -99999
            # And we might have an outgame subclass
            choices = [0, 1, 2, -99999]
            if self.mode == 2:
                choices = [0, 1, 2]
            l["y_subclass_if_outgame"] = np.random.choice(choices)
            if l["y_subclass_if_outgame"] > -1:
                x[2] += l["y_subclass_if_outgame"] - 1
        l.update({"images": x})
        return l

    def __len__(self):
        return self.len


if __name__ == "__main__":
    # model = TestPLModule()
    # ld_train = tud.DataLoader(
    #     FakeDataset(20), batch_size=3, shuffle=True, drop_last=True
    # )
    # ld_valid = tud.DataLoader(
    #     FakeDataset(20), batch_size=10, shuffle=False, drop_last=False
    # )
    # trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=5)
    # trainer.fit(model, ld_train, ld_valid)

    comet_logger = CometLogger(
        api_key="rEy6sPPJyZ6EATCASeGFbeNB8",
        project_name="whichgame-tests",
        experiment_name="multiple valid loaders",
    )

    model = WhichGameTrainer("mobilenet_v2", 4, 3)

    ld_train = tud.DataLoader(
        LessFakeDataset(16 * 128), batch_size=16, shuffle=True, drop_last=True
    )
    ld_valid = tud.DataLoader(
        LessFakeDataset(1000, 1), batch_size=16, shuffle=False, drop_last=False
    )
    ld_valid_2 = tud.DataLoader(
        LessFakeDataset(250, 2), batch_size=16, shuffle=False, drop_last=False
    )
    n_gpus = 1
    trainer = pl.Trainer(
        gpus=n_gpus if torch.cuda.is_available() else 0,
        max_epochs=15,
        accelerator="ddp",
        logger=comet_logger,
    )
    trainer.fit(model, ld_train, [ld_valid, ld_valid_2])
