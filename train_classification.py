import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_processing import datasets
from pointnet import attacks
from pointnet.model import PointNet
from util import logging
from util.math import set_random_seed, DEFAULT_SEED

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True, help='path of output directory')
parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use', choices=['modelnet40'])
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=1024, help='number of points per point cloud')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for random number generator')
parser.add_argument('--ignore_existing_output_dir', action='store_true', help='ignore if output dir exists')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel data loader workers')
parser.add_argument('--defense', action='store_true', help='use adversarial training')
parser.add_argument('--eps', type=float, default=0.02, help='radius of eps-box to defend around point')
parser.add_argument('--step_size', type=float, default=None, help='step size for FGSM')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--max_features', type=int, default=1024, help='the number of features for max pooling')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='global pooling function')
parser.add_argument('--domain', choices=['box', 'face'], default='box', help='The attack model domain')
parser.add_argument('--rotation', choices=['none', 'z', 'so3'], default='z', help='Axis for rotation augmentation')

settings = parser.parse_args()

settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
settings.out = os.path.join('out', settings.out)
if not settings.step_size:
    settings.step_size = 1.25 * settings.eps

os.makedirs(settings.out, exist_ok=settings.ignore_existing_output_dir)

log_name = f"train_defended[{settings.defense}]_eps[{settings.eps}]_rotation[settings.rotation]_pooling[{settings.pooling}]"
logger = logging.create_logger(settings.out, log_name)

logger.info(settings)

writer = SummaryWriter(log_dir=settings.out)

set_random_seed(settings.seed)

train_data = datasets.modelnet40(num_points=settings.num_points, split='train', rotate=settings.rotation)
test_data = datasets.modelnet40(num_points=settings.num_points, split='test', rotate='none')

train_loader = DataLoader(
    dataset=train_data,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=settings.num_workers
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=settings.batch_size,
    shuffle=False,
    num_workers=settings.num_workers
)

print("Train Size: ", len(train_data))
print("Test Size: ", len(test_data))
print("Total Size: ", len(test_data) + len(train_data))

distribution = np.zeros(40, dtype=int)
for sample in train_data:
    _, _, label = sample
    distribution[label.item()] += 1
print(distribution)

num_batches = len(train_data) / settings.batch_size
logger.info("Number of batches: %d", num_batches)
logger.info("Number of classes: %d", train_data.num_classes)
logger.info("Training set size: %d", len(train_data))
logger.info("Test set size: %d", len(test_data))

model = PointNet(
    number_points=settings.num_points,
    num_classes=train_data.num_classes,
    max_features=settings.max_features,
    pool_function=settings.pooling
)
model = model.to(settings.device)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

logger.info("starting training")

for epoch in range(settings.epochs):

    train_correct = 0
    train_amount = 0
    train_loss = 0

    for i, data in enumerate(tqdm(train_loader)):
        points, faces, label = data
        label: torch.Tensor = torch.squeeze(label)
        points: torch.Tensor = points.float().to(settings.device)
        faces: torch.Tensor = faces.float().to(settings.device)
        label: torch.Tensor = label.to(settings.device)

        if settings.defense:
            if settings.domain == "box":
                domain = attacks.EpsBox(points, settings.eps)
            elif settings.domain == "face":
                domain = attacks.FaceBox(faces)
            else:
                assert False, f"Unsupported domain {settings.domain}"

            model.eval()
            points = domain.random_point()
            points = attacks.fgsm(model, points, label, step_size=settings.step_size)
            points = domain.project(points)

        model.train()
        optimizer.zero_grad()

        predictions = model(points)
        loss = objective(predictions, label)
        loss.backward()
        optimizer.step()

        max_predictions = predictions.data.max(1)[1]
        correct = max_predictions.eq(label.data).cpu().sum()
        train_correct += correct.item()
        train_amount += points.size()[0]
        train_loss += loss.item()

    test_correct = 0
    test_amount = 0
    test_loss = 0

    for i, data in enumerate(test_loader):
        points, _, label = data
        points = points[:, : settings.num_points, :]
        label = torch.squeeze(label)
        points = points.to(settings.device)
        label = label.to(settings.device)

        model = model.eval()
        predictions = model(points)
        loss = objective(predictions, label)
        max_predictions = predictions.data.max(1)[1]
        correct = max_predictions.eq(label.data).cpu().sum()
        test_correct += correct.item()
        test_amount += points.size()[0]
        test_loss += loss.item()

    logger.info(
        "Epoch {epoch}: train loss: {train_loss}, train accuracy: {train_accuracy}, test loss: {test_loss}, test accuracy: {test_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_correct / train_amount,
            test_loss=test_loss,
            test_accuracy=test_correct / test_amount
        )
    )

    writer.add_scalar('accuracy/train', train_correct / train_amount, epoch)
    writer.add_scalar('loss/train', train_loss / train_amount, epoch)
    writer.add_scalar('accuracy/test', test_correct / test_amount, epoch)
    writer.add_scalar('loss/test', test_loss / test_amount, epoch)

    scheduler.step()

torch.save(model.state_dict(), os.path.join(settings.out, "model.pth"))
logger.info("finished training")
