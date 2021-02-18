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


def train_fn(model, optimizer, loss_fn):
    mean_loss = []
    x = torch.rand(1, 3, 448, 448)
    y = torch.rand(1, 7, 7, 30)
    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    print(out)
    print(out.size())
    loss = loss_fn(out, y)
    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

train_fn(model,optimizer,loss_fn)
