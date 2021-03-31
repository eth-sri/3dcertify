import argparse

import numpy as np
import onnx
import torch
from tqdm import tqdm

from data_processing import datasets
from pointnet.segmentation_model import PointNetSegmentation
from relaxations.deepg_bounds import load_spec
from transformations.rotation import RotationZ
from util import onnx_converter
from util.argparse import absolute_path
from util.experiment import Experiment
from util.math import logits_to_category, DEFAULT_SEED
from util.timing import Timer
from verifier.eran_verifier import EranVerifier

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=absolute_path, required=True, help="Path to the model to verify (.pth)")
parser.add_argument('--num_points', type=int, default=1024, help="Number of points per point cloud")
parser.add_argument('--spec-dir', type=absolute_path, help="Path to directory with DeepG specs")
parser.add_argument('--pooling', choices=['improved_max', 'max'], default='improved_max', help='The pooling function to use')
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], help='The dataset to use')
parser.add_argument('--max_features', type=int, default=1024, help='The number of global features')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for random number generator')
parser.add_argument('--experiment', type=str, help='name of the experiment')

settings = parser.parse_args()

experiment = Experiment(settings)
logger = experiment.logger

checkpoints = experiment.load_checkpoints()

test_data = datasets.shapenet(num_points=settings.num_points, split='test', rotate='none')
num_total_classes = 50

torch_model = PointNetSegmentation(
    number_points=settings.num_points,
    num_seg_classes=num_total_classes,
    encode_onnx=True
)
torch_model.load_state_dict(torch.load(settings.model, map_location=torch.device('cpu')))
torch_model = torch_model.eval()

export_file = settings.model.with_suffix('.onnx')
onnx_model = onnx_converter.convert(torch_model, settings.num_points, export_file)
logger.info(onnx.helper.printable_graph(onnx_model.graph))

eran = EranVerifier(onnx_model)
test_samples = len(test_data)
interval = len(test_data) // 100

total_percentage_certified = 0.0
iterations = 0

timer = Timer()

for counter, i in enumerate(range(0, test_samples, interval)):

    iterations += 1
    np_points, label = test_data[i]
    points = torch.from_numpy(np_points)
    points = torch.unsqueeze(points, dim=0)

    assert np_points.shape[0] == settings.num_points, \
        f"invalid points shape {np_points.shape}, expected ({settings.num_points}, x)"

    prediction = torch_model(points.transpose(2, 1)).detach().cpu().numpy()[0]
    predicted_parts = logits_to_category(prediction, label)

    correct_predicted_indexes = np.argwhere(predicted_parts == label).squeeze()
    correct_predicted_labels = label[correct_predicted_indexes]

    deepg_spec = load_spec(settings.spec_dir, counter)

    checkpoints_sample = checkpoints[str(i)] if str(i) in checkpoints else {}

    valid_classes = np.unique(label)
    certified_points = np.full(settings.num_points, True)

    timer.start()

    progress_bar = tqdm(deepg_spec, desc=f"Object {counter}", unit="interval")
    for params, bounds, constraints in progress_bar:

        interval_key = 'x'.join([f"[{i.lower_bound:.4f},{i.upper_bound:.4f}]" for i in params])
        if interval_key in checkpoints_sample:
            interval_certified = np.array(checkpoints_sample[interval_key])
        else:
            transformation = RotationZ()
            transformed_points = transformation.transform(np_points, [(i.lower_bound + i.upper_bound) / 2.0 for i in params])

            assert np.min(bounds.upper_bound - transformed_points.flatten()) >= -0.0001, "Upper bound violation!"
            assert np.min(transformed_points.flatten() - bounds.lower_bound) >= -0.0001, "Lower bound violation!"

            (dominant_classes, nlb, nub) = eran.analyze_segmentation_linear(bounds, constraints, params, label, valid_classes, num_total_classes)
            dominant_classes = np.array(dominant_classes)

            assert np.all(np.logical_or(label == dominant_classes, dominant_classes == -1)), \
                f"Wrong dominant class! label {label}, dominant_class: {dominant_classes}"

            interval_certified = dominant_classes == label
            checkpoints_sample[interval_key] = interval_certified.tolist()

        certified_points = np.logical_and(certified_points, interval_certified)

    elapsed = timer.stop()

    checkpoints[str(i)] = checkpoints_sample
    experiment.store_checkpoints(checkpoints)

    correct_predicted_certified = certified_points[correct_predicted_indexes]

    percentage_certified = np.mean(correct_predicted_certified.astype(float))
    total_percentage_certified += percentage_certified

    logger.info(f"Successfully certified {percentage_certified} of points")

    logger.info(f"Time for this round: {elapsed}s. Total time: {timer.get()}s.")
    logger.info(f"Tested {iterations} data points for which on average {total_percentage_certified / iterations} of points were correctly predicted.")
