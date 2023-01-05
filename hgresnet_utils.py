import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp

class ADE20k(Dataset):
    def __init__(self, data_path, transform, label_transform, train_val):
        assert train_val in ["training", "validation"]
        self.img_paths = sorted(glob.glob(os.path.join(data_path, f"images/{train_val}/*")))
        self.label_paths = sorted(glob.glob(os.path.join(data_path, f"annotations/{train_val}/*")))
        self.transform = transform
        self.label_transform = label_transform
        assert len(self.img_paths) == len(self.label_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        label = Image.open(self.label_paths[index])
        img = self.transform(img)
        label = self.label_transform(label)
        return img, label

class LitModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.aux_coef = 0.4
        self.loss_function = smp.losses.DiceLoss(mode="multiclass", classes = range(151), from_logits=True)

    def forward(self, x):
        out1, out2 = self.model(x)
        return out1, out2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self.model(imgs)
        loss_aux = self.loss_function(output[1], labels.long())
        loss = self.loss_function(output[0], labels.long())
        self.log("train_loss", loss + self.aux_coef*loss_aux, prog_bar=False, on_step=False, on_epoch=True)
        return loss + self.aux_coef*loss_aux

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self.model(imgs)
        loss_aux = self.loss_function(output[1], labels.long())
        loss = self.loss_function(output[0], labels.long())
        self.log("valid_loss", loss + self.aux_coef*loss_aux, prog_bar=False, on_step=False, on_epoch=True)

        pred_mask = torch.argmax(output, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, labels.long(), mode="multiclass", num_classes=151)
        micro_mean_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        macro_mean_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log("micro_mIoU", micro_mean_iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("macro_mIoU", macro_mean_iou, prog_bar=False, on_step=False, on_epoch=True)
        return loss