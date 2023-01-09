import os
import torch
import numpy as np
import pytorch_lightning as pl
from hgresnet_utils import ADE20k, LitModel
from hgresnet import HGResNet
from torchvision import transforms as T
from torch.utils.data import DataLoader

BATCH_SIZE = 16
transform = T.Compose([T.Resize((473,473)),
                       T.Lambda(lambda x: x.convert("RGB")),
                       T.ToTensor()])
label_transform = T.Compose([T.Resize((473,473)),
                             T.Lambda(lambda x: torch.tensor(np.array(x)))
                             ])
train_dataset = ADE20k("ADEChallengeData2016", transform, label_transform, "training")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, persistent_workers=False)

valid_dataset = ADE20k("ADEChallengeData2016", transform, label_transform, "validation")
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, persistent_workers=False)

# データセットの形式の都合上、クラス数は151とし、index=0を無視するように実装
model = HGResNet((473,473), 151, 16).cuda()
lit_model = LitModel(model)
trainer = pl.Trainer(
    enable_progress_bar=True,
    max_epochs=100,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    num_sanity_val_steps=10,
    devices=1
)
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
