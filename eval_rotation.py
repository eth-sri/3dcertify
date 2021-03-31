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
from pointnet.model import PointNet
from util import rotation
from util.math import set_random_seed


def evaluate_base(model: nn.Module, points: torch.Tensor, label: torch.Tensor) -> Tuple[int, torch.Tensor]:
    (predictions, _) = model(points)
    max_predictions = predictions.data.max(1)[1]
    return max_predictions.eq(label).sum().item(), max_predictions


def evaluate_majority_vote(model: nn.Module, points: torch.Tensor, label: torch.Tensor, rounds: int) -> int:
    batch_predictions = []
    for j in range(rounds):
        theta = (j * np.pi * 2) / rounds
        rotated_points = rotation.rotate_z_batch(points, theta)
        (predictions, _) = model(rotated_points)
        max_predictions = predictions.data.max(1)[1]
        batch_predictions.append(max_predictions.cpu().numpy())

    batch_predictions = np.transpose(np.array(batch_predictions))
    votes = np.zeros((batch_predictions.shape[0], test_data.num_classes))
    for k in range(batch_predictions.shape[0]):
        for j in range(batch_predictions.shape[1]):
            votes[k][batch_predictions[k][j]] += 1
    majority = np.argmax(votes, axis=1)
    return np.equal(majority, label.cpu().numpy()).sum()


def evaluate_so3(model: nn.Module, points: torch.Tensor, label: torch.Tensor) -> int:
    points = rotation.random_rotate_so3_batch(points)
    (predictions, _) = model(points)
    max_predictions = predictions.data.max(1)[1]
    return max_predictions.eq(label).sum().item()


def evaluate_z(model: nn.Module, points: torch.Tensor, label: torch.Tensor) -> int:
    points = rotation.random_rotate_z_batch(points)
    (predictions, _) = model(points)
    max_predictions = predictions.data.max(1)[1]
    return max_predictions.eq(label).sum().item()


def evaluate_random_attack_z(model: nn.Module, points: torch.Tensor, label: torch.Tensor, theta: float, iterations: int):
    adversarial_samples = torch.zeros_like(label, dtype=torch.long)
    for i in range(iterations):
        rotated_points = rotation.random_rotate_z_batch(points, -theta, theta)
        (predictions, _) = model(rotated_points)
        max_predictions = predictions.data.max(1)[1]
        adversarial_samples += (max_predictions != label)
    return adversarial_samples.size(0) - (adversarial_samples > 0).sum().item()


def evaluate_random_attack_so3(model: nn.Module, points: torch.Tensor, label: torch.Tensor, theta: float, iterations: int):
    adversarial_samples = torch.zeros_like(label, dtype=torch.long)
    for i in range(iterations):
        rotated_points = rotation.random_rotate_so3_batch(points, -theta, theta)
        (predictions, _) = model(rotated_points)
        max_predictions = predictions.data.max(1)[1]
        adversarial_samples += (max_predictions != label)
    return adversarial_samples.size(0) - (adversarial_samples > 0).sum().item()


def evaluate_grid_z(model: nn.Module, points: torch.Tensor, label: torch.Tensor, theta: float, iterations: int):
    theta_min = -theta
    theta_delta = (2 * theta) / iterations
    adversarial_samples = torch.zeros_like(label, dtype=torch.long)
    for i in range(iterations + 1):
        rotated_points = rotation.rotate_z_batch(points, theta_min + i * theta_delta)
        (predictions, _) = model(rotated_points)
        max_predictions = predictions.data.max(1)[1]
        adversarial_samples += (max_predictions != label)
    return adversarial_samples.size(0) - (adversarial_samples > 0).sum().item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='path to the trained model')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points per point cloud')
    parser.add_argument('--num_workers', type=int, default=0, help='number of parallel data loader workers')
    parser.add_argument('--eval_rotations', type=int, default=12, help='amount of rotations to evaluate')
    parser.add_argument('--theta', type=float, default=1, help='angle to attack')
    parser.add_argument('--best_of', type=int, default=10, help='best of k for random attack')
    parser.add_argument('--max_features', type=int, default=1024, help='the number of features for max pooling')
    parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='global pooling function')
    parser.add_argument('--seed', type=int, default=182343073, help='seed for random number generator')

    settings = parser.parse_args()

    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.dataset = os.path.join('data', settings.dataset)

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

    theta = settings.theta / 180.0

    num_correct_base = 0
    num_correct_vote = 0
    num_correct_z = 0
    num_correct_so3 = 0
    num_correct_rand_z = 0
    num_correct_grid_z = 0
    num_correct_rand_so3 = 0
    num_total = 0

    for i, data in enumerate(tqdm(test_loader)):
        points, _, label = data
        label = torch.squeeze(label)
        points = points.to(settings.device)
        label = label.to(settings.device)

        correct, _ = evaluate_base(model, points, label)
        num_correct_base += correct
        num_correct_vote += evaluate_majority_vote(model, points, label, settings.eval_rotations)
        num_correct_z += evaluate_z(model, points, label)
        num_correct_so3 += evaluate_so3(model, points, label)
        num_correct_grid_z += evaluate_grid_z(model, points, label, theta, settings.best_of)
        num_correct_rand_z += evaluate_random_attack_z(model, points, label, theta, settings.eval_rotations)
        num_correct_rand_so3 += evaluate_random_attack_so3(model, points, label, theta, settings.eval_rotations)

        num_total += len(label)

    logger.info("Test Accuracies:")
    logger.info(f"Base: {num_correct_base / num_total}")
    logger.info(f"Voted: {num_correct_vote / num_total}")
    logger.info(f"Z: {num_correct_z / num_total}")
    logger.info(f"SO3: {num_correct_so3 / num_total}")
    logger.info(f"AdvRandZ: {num_correct_rand_z / num_total}")
    logger.info(f"AdvGridZ: {num_correct_grid_z / num_total}")
    logger.info(f"AdvRandSO3: {num_correct_rand_so3 / num_total}")
