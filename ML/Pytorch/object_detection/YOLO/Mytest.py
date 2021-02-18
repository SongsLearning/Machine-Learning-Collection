import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import numpy as np
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0

DEVICE = "cuda" if torch.cuda.is_available else "cpu"

model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
loss_fn = YoloLoss()


IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
NUM_WORKERS = 0
BATCH_SIZE = 1
PIN_MEMORY = True

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

test_dataset = VOCDataset(
    "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
)

test_dataset.__getitem__(0)

test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
def train_fn(train_loader, model, optimizer, loss_fn):

    mean_loss = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # print(x.size())
        # print(y.size())

        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        # print(out)
        # print(out.size())
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

for i in range(300):
    train_fn(test_loader, model,optimizer,loss_fn)
