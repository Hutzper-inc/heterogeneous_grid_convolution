import os
import torch
import numpy as np
import pytorch_lightning as pl
from hgresnet_utils import ADE20k, LitModel
from hgresnet import HGResNet
from torchvision import transforms as T
from torch.utils.data import DataLoader

transform = T.Compose([T.Resize((473,473)),
                       T.Lambda(lambda x: x.convert("RGB")),
                       T.ToTensor()])
label_transform = T.Compose([T.Resize((60,60)),
                             T.Lambda(lambda x: torch.tensor(np.array(x)))
                             ])
train_dataset = ADE20k("../../validation_dataset/ADEChallengeData2016", transform, label_transform, "training")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, persistent_workers=False)

valid_dataset = ADE20k("../../validation_dataset/ADEChallengeData2016", transform, label_transform, "validation")
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, persistent_workers=False)

model = HGResNet((473,473), 151, 32).cuda()
lit_model = LitModel(model)
trainer = pl.Trainer(
    enable_progress_bar=True,
    max_epochs=100,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    num_sanity_val_steps=10,
    devices=1
)
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
