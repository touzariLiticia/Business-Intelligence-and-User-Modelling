import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from network import Net
from xent import cross_entropy


def accuracy(logit, y):
    pred = logit.argmax(dim=1, keepdim=True)
    return pred.eq(y.view_as(pred)).float().mean()


class WhichGameTrainer(Net, pl.LightningModule):
    def __init__(
        self,
        backbone_name,
        num_classes,
        num_outgame_subclasses,
        dropout_fts,
        pretrained,
        lr,
        epochs,
        loss_ingame_unld_wt=0.0,
        ingame_conf_thresh=0.1,
        loss_ingame_unld_minent_wt=0.0,
        whichgame_smooth_eps=0.0,
        ingame_smooth_eps=0.0,
        loss_weights={"loss_outgame_subclasses": 5.0},
    ):
        super().__init__(
            backbone_name,
            num_classes,
            num_outgame_subclasses,
            dropout_fts,
            pretrained,
        )
        self.lr = lr
        self.epochs = epochs
        self.loss_ingame_unld_wt = loss_ingame_unld_wt
        self.ingame_conf_thresh = ingame_conf_thresh
        self.loss_ingame_unld_minent_wt = loss_ingame_unld_minent_wt
        self.whichgame_smooth_eps = whichgame_smooth_eps
        self.ingame_smooth_eps = ingame_smooth_eps
        self.loss_weights = loss_weights

    def _write_to_log(self, losses, pre_str):
        for k, (v, length) in losses.items():
            self.log(f"{pre_str}{k}", v)

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        return self.calc_losses(batch, calc_accuracies=True)

    def _validation_epoch_end(self, outputs):
        agg_dict = {}
        for k in outputs[0]:
            sum_lens = sum([d[k][1] for d in outputs if k in d])
            if sum_lens > 0:
                agg_dict[k] = (
                    sum(
                        [
                            d[k][0] * d[k][1] / sum_lens
                            for d in outputs
                            if k in d
                        ]
                    ),
                    1,
                )
        return agg_dict

    def validation_epoch_end(self, outputs_per_dataloader):
        agg_dicts = []
        for i_loader, outputs in enumerate(outputs_per_dataloader):
            agg_dict = self._validation_epoch_end(outputs)
            agg_dicts.append(agg_dict)
            self._write_to_log(agg_dict, f"valid{i_loader}_")
            print(f"Loader {i_loader} " + str(agg_dict))

    def training_step(self, batch, batch_idx):
        losses = self.calc_losses(batch)
        self._write_to_log(losses, "train_")
        return losses["loss"][0]

    def _calc_overall_loss_from_losses(self, losses, loss=None):
        for k, (v, length) in losses.items():
            if k.startswith("_"):
                continue
            if loss is None:
                loss = v
            else:
                loss += v
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=0.007, momentum=0.9, nesterov=True,
        # )
        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.lr / 100
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_scheduler",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def calc_losses(self, batch, calc_accuracies=False):
        x = batch["images"]
        (
            logit_ingame,
            logit_whichgame,
            logit_outgame_subclasses,
            ftrs_for_distill,
            ftrs,
        ) = self(x)

        # Figure out which batch elements have which types of labels
        ingame_lbld = batch["y_ingame"] != -99999
        ingame_unld = batch["y_ingame"] == -99999
        outgame_subclass_lbld = (
            batch["y_subclass_if_outgame"] != -99999
            if "y_subclass_if_outgame" in batch
            else [0]
        )
        whichgame_lbld = (batch["y_ingame"] == 1) & (
            batch["y_whichgame"] != -99999
        )
        whichgame_unld = (batch["y_ingame"] == -99999) & (
            batch["y_whichgame"] != -99999
        )

        # Calculate the loss components
        losses = {}

        # Elements where we have definite labels for in- vs. out-game
        if sum(ingame_lbld):
            losses["loss_ingame_lbld"] = (
                cross_entropy(
                    logit_ingame[ingame_lbld],
                    batch["y_ingame"][ingame_lbld],
                    smooth_eps=self.ingame_smooth_eps,
                ),
                len(batch["y_ingame"][ingame_lbld]),
            )
            if calc_accuracies:
                losses["_accu_ingame_lbld"] = (
                    accuracy(
                        logit_ingame[ingame_lbld],
                        batch["y_ingame"][ingame_lbld],
                    ),
                    len(batch["y_ingame"][ingame_lbld]),
                )

        # Definitely-outgame elements where we have an outgame subclass label
        if sum(outgame_subclass_lbld):
            # We may have subclasses for outgame like NSFW, porn, etc.
            losses["loss_outgame_subclasses"] = (
                F.cross_entropy(
                    logit_outgame_subclasses[outgame_subclass_lbld],
                    batch["y_subclass_if_outgame"][outgame_subclass_lbld],
                )
                * self.loss_weights.get("loss_outgame_subclasses", 1.0),
                len(logit_outgame_subclasses[outgame_subclass_lbld]),
            )
            if calc_accuracies:
                losses["_accu_outgame_subclasses"] = (
                    accuracy(
                        logit_outgame_subclasses[outgame_subclass_lbld],
                        batch["y_subclass_if_outgame"][outgame_subclass_lbld],
                    ),
                    len(logit_outgame_subclasses[outgame_subclass_lbld]),
                )

        # Definitely-ingame elements where we have a class label
        if sum(whichgame_lbld):
            losses["loss_whichgame_lbld"] = (
                F.cross_entropy(
                    logit_whichgame[whichgame_lbld],
                    batch["y_whichgame"][whichgame_lbld],
                ),
                len(logit_whichgame[whichgame_lbld]),
            )
            if calc_accuracies:
                losses["_accu_whichgame_lbld"] = (
                    accuracy(
                        logit_whichgame[whichgame_lbld],
                        batch["y_whichgame"][whichgame_lbld],
                    ),
                    len(logit_whichgame[whichgame_lbld]),
                )

        # Intermediate calculation of p(ingame) required for pseudo-labeling
        if sum(ingame_unld) or sum(whichgame_unld):
            sm_ingame = torch.softmax(logit_ingame, dim=1)
            prob_ingame = sm_ingame[:, 1].detach()
            losses["_prob_ingame_unld_avg"] = (
                prob_ingame[ingame_unld].mean(),
                len(prob_ingame[ingame_unld]),
            )

        # Some unsupervised loss terms for unlabled in/outgame examples
        if sum(ingame_unld):
            if self.loss_ingame_unld_wt:
                # Pseudo-label confident ingame predictions
                conf_ingame = (
                    prob_ingame[ingame_unld] < self.ingame_conf_thresh
                ) | (prob_ingame[ingame_unld] > (1 - self.ingame_conf_thresh))
                if sum(conf_ingame):
                    pseudolabels_ingame = (
                        prob_ingame[ingame_unld][conf_ingame] > 0.5
                    ).long()
                if sum(conf_ingame):
                    loss_ingame_unld = (
                        F.cross_entropy(
                            logit_ingame[ingame_unld][conf_ingame],
                            pseudolabels_ingame,
                        ),
                        len(pseudolabels_ingame),
                    )
                    losses["loss_ingame_unld"] = (
                        (self.loss_ingame_unld_wt * loss_ingame_unld),
                        len(pseudolabels_ingame),
                    )
                    losses["sum_conf_ingame"] = (
                        sum(conf_ingame),
                        len(conf_ingame),
                    )
            # Minimise entropy of in/outgame predictions
            if self.loss_ingame_unld_minent_wt:
                losses["loss_ingame_unld_minent"] = (
                    (
                        self.loss_ingame_unld_minent_wt
                        * entropy(sm_ingame[ingame_unld])
                    ),
                    len(sm_ingame[ingame_unld]),
                )

        # Smoothed cross-ent for images with weak which-game label (no ingame)
        if sum(whichgame_unld):
            losses["loss_whichgame_unld"] = (
                (
                    cross_entropy(
                        logit_whichgame[whichgame_unld],
                        batch["y_whichgame"][whichgame_unld],
                        smooth_eps=self.whichgame_smooth_eps,
                        sample_weight=prob_ingame[whichgame_unld],
                        reduction="sum",
                    )
                    / sum(whichgame_unld)
                    * (10 if self.whichgame_smooth_eps else 1)
                ),
                len(batch["y_whichgame"][whichgame_unld]),
            )
            if calc_accuracies:
                losses["_accu_whichgame_unld"] = (
                    accuracy(
                        logit_whichgame[whichgame_unld],
                        batch["y_whichgame"][whichgame_unld],
                    ),
                    len(batch["y_whichgame"][whichgame_unld]),
                )

        losses["loss"] = self._calc_overall_loss_from_losses(losses), len(x)
        return losses
