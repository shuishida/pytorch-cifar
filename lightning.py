'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from data import get_test_loader, get_train_loader

from utils import progress_bar
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from models import models


class Classify(pl.LightningModule):
    def __init__(self, model="resnet", lr=0.1):
        super().__init__()
        self.lr = lr
        self.model = models[model]
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def _step(self, batch, batch_idx, tag):
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        self.log(f"{tag}_acc", acc, on_epoch=True)
        return {"loss": loss, "pred": pred, "acc": acc}

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        acc = torch.stack([o['acc'] for o in outputs]).mean().item()
        print(acc)

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
                           
        return [self.optimizer], [self.scheduler]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default="resnet18", type=str, help='select model')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()

    batch_size = 128
    n_workers = 4

    train_loader = get_train_loader()
    val_loader = get_test_loader()
    
    module = Classify(args.model, args.lr)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", gpus=1, callbacks=[lr_monitor])
    trainer.fit(module, train_loader, val_loader)
