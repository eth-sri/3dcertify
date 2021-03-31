import argparse

import numpy as np
import torch
from auto_LiRPA.perturbations import *
from tqdm import tqdm

from auto_LiRPA import BoundedModule, BoundedTensor
from data_processing import datasets
from lirpa_integration import SemanticTransformation
from pointnet.model import PointNet
from relaxations.interval import Interval
from transformations.rotation import RotationZ
from util.argparse import parse_theta, absolute_path
from util.experiment import Experiment
from util.math import set_random_seed, DEFAULT_SEED

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=absolute_path, required=True, help="Path to the model to verify (.pth)")
parser.add_argument('--num_points', type=int, default=1024, help="Number of points per point cloud")
parser.add_argument('--theta', type=parse_theta, required=True, help="List of transformation parameters to certify. Either number or number followed by 'deg' for degree.")
parser.add_argument('--max_features', type=int, default=1024, help='The number of global features')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for random number generator')
parser.add_argument('--batch-size', type=int, default=10, help="Certification batch size")
parser.add_argument('--experiment', type=str, help='name of the experiment')

settings = parser.parse_args()

experiment = Experiment(settings)
logger = experiment.logger

set_random_seed(settings.seed)

test_data = datasets.modelnet40(num_points=settings.num_points, split='test', rotate='none')

torch_model = PointNet(
    number_points=settings.num_points,
    num_classes=test_data.num_classes,
    max_features=settings.max_features,
    pool_function='max',
    disable_assertions=True
)
torch_model.load_state_dict(torch.load(settings.model, map_location=torch.device('cpu')))

torch_model = torch_model.eval()

lirpa_model = BoundedModule(torch_model, torch.empty(2, settings.num_points, 3))
lirpa_model.eval()

theta = settings.theta
perturbation = SemanticTransformation(RotationZ(), [Interval(-theta, theta)])

test_samples = len(test_data)
interval = len(test_data) // 100

num_class = 40

correct_predictions = 0
verified_same = 0
verified_different = 0
not_verified = 0
iterations = 0

total_time = 0

for i in tqdm(range(0, test_samples, interval)):
    iterations += 1
    np_points, faces, true_label = test_data[i]

    points = torch.from_numpy(np_points).unsqueeze(0)

    relaxed_points = BoundedTensor(points, perturbation)

    torch_model.eval()
    prediction = torch_model(points)[0].squeeze()
    max_prediction = prediction.data.max(0)[1].item()
    correct = max_prediction == true_label

    lirpa_model.eval()
    lirpa_prediction = lirpa_model(relaxed_points)[0].squeeze()

    assert torch.all(lirpa_prediction == prediction), f"Lirpa prediction differs from torch prediction! Torch: {prediction}, LiRPA: {lirpa_prediction}"

    if not correct:
        logger.info("Incorrect prediction, skipping. True label was {}, prediction was {}".format(true_label, max_prediction))
        continue

    correct_predictions += 1

    labels = torch.tensor([true_label])
    c = torch.eye(num_class).type_as(points)[labels].unsqueeze(1) - torch.eye(num_class).type_as(points).unsqueeze(0)
    # remove specifications to self
    I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
    c = (c[I].view(points.size(0), num_class - 1, num_class))

    ## Step 5: Compute bounds for final output
    method = 'backward'
    lb, ub = lirpa_model.compute_bounds(x=(relaxed_points,), method=method, C=c, bound_upper=False)
    lb = lb.detach().cpu().numpy().squeeze()
    prediction = prediction.detach().cpu().numpy()

    certified = np.all(lb >= 0)

    if certified:
        logger.info(f"Successfully verified class {true_label}")
        verified_same += 1
    else:
        logger.info(f"Failed to verify class {true_label}")
        not_verified += 1

    logger.info(f"Tested {iterations} data points out of which {correct_predictions} were correctly predicted.")
    logger.info(f"Successfully verified {verified_same} samples, unable to verify {not_verified} samples. " +
                f"{verified_different} were falsely verified as a different class.")
