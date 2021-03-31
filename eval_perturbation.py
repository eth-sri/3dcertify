import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing import datasets
from pointnet import attacks
from pointnet.model import PointNet
from util import rotation
from util.math import set_random_seed


def evaluate_base(model: nn.Module, points: torch.Tensor, label: torch.Tensor) -> Tuple[int, torch.Tensor]:
    predictions = model(points)
    max_predictions = predictions.data.max(1)[1]
    return max_predictions.eq(label).sum().item(), max_predictions


def evaluate_majority_vote(model: nn.Module, points: torch.Tensor, label: torch.Tensor, rounds: int) -> int:
    batch_predictions = []
    for j in range(rounds):
        theta = (j * np.pi * 2) / rounds
        rotated_points = rotation.rotate_z_batch(points, theta)
        predictions = model(rotated_points)
        max_predictions = predictions.data.max(1)[1]
        batch_predictions.append(max_predictions.cpu().numpy())

    batch_predictions = np.transpose(np.array(batch_predictions))
    votes = np.zeros((batch_predictions.shape[0], test_data.num_classes))
    for k in range(batch_predictions.shape[0]):
        for j in range(batch_predictions.shape[1]):
            votes[k][batch_predictions[k][j]] += 1
    majority = np.argmax(votes, axis=1)
    return np.equal(majority, label.cpu().numpy()).sum()


def evaluate_bgd(model: nn.Module, domain: attacks.Domain, label: torch.Tensor, eps_step: float, fgsm_iter: int) -> \
        Tuple[int, torch.Tensor]:
    adversarial_points = attacks.pgd(model, domain, label, fgsm_iter, eps_step)
    predictions = model(adversarial_points)
    max_predictions = predictions.data.max(1)[1]
    return max_predictions.eq(label).sum().item(), max_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='path to the trained model')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points per point cloud')
    parser.add_argument('--num_workers', type=int, default=0, help='number of parallel data loader workers')
    parser.add_argument('--eval_rotations', type=int, default=12, help='amount of rotations to evaluate')
    parser.add_argument('--eps', type=float, default=0.01, help='radius of box around points to attack')
    parser.add_argument('--eps_step', type=float, default=None, help='step size of pgd attack, default is eps/2')
    parser.add_argument('--fgsm_iter', type=int, default=50, help='iterations of fgsm for pgd attack')
    parser.add_argument('--max_features', type=int, default=1024, help='the number of features for max pooling')
    parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='global pooling function')
    parser.add_argument('--domain', choices=['box', 'face'], default='box', help='attack model domain')
    parser.add_argument('--seed', type=int, default=18253073, help='seed for random number generator')

    settings = parser.parse_args()

    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.dataset = os.path.join('data', settings.dataset)
    if not settings.eps_step:
        settings.eps_step = 0.5 * settings.eps

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    logger.info(settings)

    set_random_seed(settings.seed)

    test_data = datasets.modelnet40(num_points=settings.num_points, split='test', rotate='none')

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers
    )

    num_batches = len(test_data) / settings.batch_size
    logger.info("Number of batches: %d", num_batches)
    logger.info("Number of classes: %d", test_data.num_classes)
    logger.info("Test set size: %d", len(test_data))

    model = PointNet(
        number_points=settings.num_points,
        num_classes=test_data.num_classes,
        max_features=settings.max_features,
        pool_function=settings.pooling
    )
    model.load_state_dict(torch.load(settings.model))
    model = model.to(settings.device)
    model = model.eval()

    logger.info("starting evaluation")

    num_correct_base = 0
    num_correct_vote = 0
    num_correct_pgd = 0
    num_total = 0

    distribution = np.zeros(40)
    confusion_matrix = np.zeros((40, 40))
    adv_confusion_matrix = np.zeros((40, 40))

    for i, data in enumerate(tqdm(test_loader)):
        points, faces, label = data
        label = torch.squeeze(label)
        for l in label:
            distribution[l.item()] += 1
        points = points.to(settings.device)
        faces = faces.to(settings.device)
        label = label.to(settings.device)

        correct, predictions = evaluate_base(model, points, label)
        num_correct_base += correct
        for prediction, actual in zip(predictions, label):
            confusion_matrix[actual, prediction] += 1

        num_correct_vote += evaluate_majority_vote(model, points, label, settings.eval_rotations)

        if settings.domain == "box":
            domain = attacks.EpsBox(points, settings.eps)
        elif settings.domain == "face":
            domain = attacks.FaceBox(faces)
        else:
            assert False, f"Unsupported domain {settings.domain}"

        correct, predictions = evaluate_bgd(model, domain, label, settings.eps_step, settings.fgsm_iter)
        num_correct_pgd += correct
        for prediction, actual in zip(predictions, label):
            adv_confusion_matrix[actual, prediction] += 1

        num_total += len(label)

    logger.info(
        "Test Accuracy: Base: {base_accuracy}, Vote: {vote_accuracy}, BGD: {pgd_accuracy}".format(
            base_accuracy=num_correct_base / num_total,
            vote_accuracy=num_correct_vote / num_total,
            pgd_accuracy=num_correct_pgd / num_total
        )
    )
