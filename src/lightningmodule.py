from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from architectures import ARCHS
from src.parser_sanitizers import smooth_eps_parser, str2bool


class TrainingModule(pl.LightningModule):

    def __init__(self,
                 arch: str,
                 channels: tuple,
                 num_classes: int,
                 tab_size: int,
                 labels: list,
                 smooth_eps: float,
                 lr: float,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.arch = arch
        self.channels = channels
        self.num_channels = len(self.channels)
        self.num_classes = num_classes
        self.tab_size = tab_size
        self.labels = labels
        self.smooth_eps = smooth_eps
        self.model = ARCHS[self.arch](num_classes=self.num_classes,
                                      num_channels=self.num_channels,
                                      tab_size=self.tab_size, **kwargs)
        self.loss_fn = LabelSmoothingCrossEntropy(eps=self.smooth_eps)
        self.lr = lr

        # they are put in kwargs as not to be saved in tensorboard
        self.scheduler_patience = kwargs["scheduler_patience"]
        self.scheduler_cooldown = kwargs["scheduler_cooldown"]
        self.save_test_preds = kwargs["save_test_preds"]

        #metrics
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(num_classes=self.num_classes)

        self.test_preds=[]

        # bullshit to determine positive class
        self.positive_class = 1
        if self.num_classes == 2:
            try:
                self.positive_class = labels.index("OK")
            except ValueError:
                try:
                    self.positive_class = labels.index("0")
                except ValueError:
                    try:
                        self.positive_class = 1 - labels.index("The rest")
                    except ValueError:
                        pass

    def forward(self, data_tuple):
        return self.model(data_tuple)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y["targets"])
        preds = F.softmax(y_hat, dim=1)
        # self.log("true_loss", self.loss_fn(y_hat, y["targets"]), on_epoch=True, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, y["targets"]), on_epoch=False, on_step=True, prog_bar=True)

        return loss

    def _eval_step(self, batch, batch_idx, prefix):
        output = {}
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y["targets"])

        preds = F.softmax(y_hat, dim=1)
        acc = self.accuracy(preds, y["targets"])
        f1 = self.f1(preds, y["targets"])
        #auroc = pl.metrics.functional.classification.multiclass_auroc(preds, y["targets"])
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
        self.log(f"{prefix}_f1", f1, prog_bar=True)
        if prefix=="test":
            self.test_preds.append(preds.cpu().numpy())
        output[f"{prefix}_loss"] = loss
        output[f"{prefix}_acc"] = acc

        return output

    def validation_step(self, batch, batch_idx):
        output = self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        output = self._eval_step(batch, batch_idx, "test")
        return output

    def on_test_epoch_end(self) -> None:
        self.test_preds = np.concatenate(self.test_preds, axis=0)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.model.scheduler is None:
            sched = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True,
                                                                        patience=self.scheduler_patience,
                                                                        mode="min",
                                                                        cooldown=self.scheduler_cooldown),
                "monitor": 'val_loss',
                "interval": 'epoch',
                "frequency": 1,
            }

        else:
            sched = {
                "scheduler": self.model.scheduler["fn"](optim, **self.model.scheduler["kwargs"]),
                "monitor": 'val_loss',
                "interval": 'epoch',
                "frequency": 1,
            }
        return [optim], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--smooth_eps",
                            type=smooth_eps_parser,
                            default=0.0,
                            metavar="[0 <= r <= 1]",
                            help=f"Indice entre 0 et 1 indiquant à quel point l'étiquette est adoussie. " \
                                 f"Example avec 0.1: Lorsque l'étiquette est (1,0,0), " \
                                 f"cette dernière est addoussie à (0.9, 0.05, 0.05). " \
                                 f"Ceci réduit le dommage que les fausses étiquettes ont sur la classification. " \
                                 f"Défaut: 0.0")
        parser.add_argument("--arch",
                            type=str,
                            default="ResNetV2",
                            choices=ARCHS.keys(),
                            help="Architecture choisie pour l'entrainement d'un nouveau modèle. Défaut: ResNetV2")

        return parser


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps: float = 0.0, reduction='mean'):
        super().__init__()
        assert 0.0 <= eps <= 1.0
        self.eps, self.reduction = eps, reduction

    def forward(self, output, target, weight=None):
        # number of classes
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        loss = reduce_loss(-log_preds.sum(dim=1), self.reduction)
        nll = F.nll_loss(log_preds, target, weight=weight, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1 - self.eps) * nll + self.eps * (loss / c)
