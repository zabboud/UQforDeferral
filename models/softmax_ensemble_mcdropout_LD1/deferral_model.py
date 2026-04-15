from ResNet50 import ResNet_50
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from AIROGS_dataloader import AIROGS
from torchmetrics.classification import  BinaryROC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class Model(pl.LightningModule):
    def __init__(self, image_channels, num_classes, type, c):
        super().__init__()

        self.save_hyperparameters()
        self.model = ResNet_50(image_channels, num_classes)
        self.type = type
        self.roc = BinaryROC().to("cpu")
        self.c = c

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if self.type == "default":
            loss = F.cross_entropy(out, y)
        else:
            loss = self.deferral_loss(out, y, self.c)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def evaluate(self, stage, batch, batch_idx):
        x, y = batch
        out = self(x)

        # loss
        if self.type == "default":
             loss = F.cross_entropy(out, y)
        else:
            loss = self.deferral_loss(out, y, self.c)
        self.log(f"{stage}/loss", loss, on_epoch=True)
        
        # pAUC
        self.roc.update(out[:,1].cpu(), y.cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.evaluate("test", batch, batch_idx)

    def on_validation_epoch_end(self):
        # log pAUC over whole dataset
        pAUC = self.compute_pAUC()
        self.log("val/pauc", pAUC, on_epoch=True)
    
    def on_test_epoch_end(self):
        # log pAUC over whole dataset
        pAUC = self.compute_pAUC()
        self.log("test/pauc", pAUC, on_epoch=True)

        fpr, tpr, _ = self.roc.compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                        estimator_name='ResNet50')
        display.plot()
        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def test_dataloader(self):
        # path = "/AIROGS.h5"
        # test_dataset = AIROGS(file_path=path, t="test", transform=None)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
        
        return DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    def deferral_loss(self, out, target, c = 0.3, eps_cst= 1e-8):
        loss = eps_cst
        batch_size = out.size(0)
        defer_class = 2

        for i in range(batch_size):
        
            l = -c * torch.log(
                torch.exp(out[i, target[i]]) / torch.sum(torch.exp(out[i]))
            ) - (1 - c) * torch.log(
                (torch.exp(out[i, target[i]]) + torch.exp(out[i, defer_class])) / torch.sum(torch.exp(out[i]))
            )
            loss += l

        return loss / batch_size

    def deferral_loss_gce(self, out, target, q = 0.7, c = 0.3, eps_cst= 1e-8):
        loss = eps_cst
        batch_size = out.size(0)
        defer_class = 2

        for i in range(batch_size):
        
            l = (c/q) * (
                    1 - (
                    torch.exp(out[i, target[i]]) / torch.sum(torch.exp(out[i]))
                    ) ** q
                ) + ((1 - c) / q) * (
                    1 - (
                        (torch.exp(out[i, target[i]]) + torch.exp(out[i, defer_class])) / torch.sum(torch.exp(out[i]))
                    ) ** q
                )
            loss += l

        return loss / batch_size
     
    def generalised_cross_entropy_loss(self, out, target, q = 0.7, eps_cst = 1e-8):
        loss = eps_cst
        batch_size = out.size(0)

        for i in range(batch_size):
            l = 1/q * (
                1 - (
                    torch.exp(out[i, target[i]]) / torch.sum(torch.exp(out[i]))
                ) ** q
            )
            loss += l
        
        return loss / batch_size

    def compute_pAUC(self, specificity_range=(0.9, 1.0)):
        fpr, tpr, _ = self.roc.compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        specificity = 1 - fpr

        mask = (specificity >= specificity_range[0]) & (specificity <= specificity_range[1])
        selected_fpr = fpr[mask]
        selected_tpr = tpr[mask]

        if len(selected_fpr) > 1:
            pAUC = metrics.auc(selected_fpr, selected_tpr)
            # normalise to [0,1] by max possible AUc in this range
            pAUC = pAUC / (specificity_range[1] - specificity_range[0])
            pAUC = torch.from_numpy(np.array(pAUC))
        else:
            pAUC = torch.tensor(0.0)

        return pAUC
    
