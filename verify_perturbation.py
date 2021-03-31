import argparse
from timeit import default_timer as timer

import numpy as np
import onnx
import onnxruntime
import torch

from data_processing import datasets
from pointnet.model import PointNet
from relaxations.interval import Interval
from util import onnx_converter
from util.argparse import absolute_path
from util.experiment import Experiment
from util.math import set_random_seed, DEFAULT_SEED
from verifier.eran_verifier import EranVerifier

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=absolute_path, required=True, help="Path to the model to verify (.pth)")
parser.add_argument('--num_points', type=int, default=1024, help="Number of points per point cloud")
parser.add_argument('--eps', type=float, default=0.01, help="Epsilon-box to certify around the input point")
parser.add_argument('--pooling', choices=['improved_max', 'max', 'avg'], default='improved_max', help='The pooling function to use')
parser.add_argument('--max_features', type=int, default=1024, help='The number of global features')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for random number generator')
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
    pool_function=settings.pooling,
    disable_assertions=True,
    transposed_input=True
)
torch_model.load_state_dict(torch.load(settings.model, map_location=torch.device('cpu')))
torch_model = torch_model.eval()

export_file = settings.model.with_suffix('.onnx')
onnx_model = onnx_converter.convert(torch_model, settings.num_points, export_file)
logger.info(onnx.helper.printable_graph(onnx_model.graph))

eran = EranVerifier(onnx_model)
test_samples = len(test_data)
interval = len(test_data) // 100

correct_predictions = 0
verified_same = 0
verified_different = 0
not_verified = 0
iterations = 0

total_time = 0

for count, i in enumerate(range(0, test_samples, interval)):

    iterations += 1
    np_points, faces, label = test_data[i]

    points = torch.from_numpy(np_points)
    points = torch.unsqueeze(points, dim=0)

    prediction = torch_model(points.transpose(2, 1))
    max_prediction = prediction.data.max(1)[1].item()
    correct = max_prediction == label

    if not correct:
        logger.info("Incorrect prediction, skipping. True label was {}, prediction was {}".format(label, max_prediction))
        continue

    correct_predictions += 1

    lower_bound = np_points - settings.eps
    upper_bound = np_points + settings.eps

    logger.info("Verifying onnx model...")
    session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    input_data = np.expand_dims(np.transpose(np_points.copy()), axis=(0, -1))
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)

    assert np.all(lower_bound <= upper_bound)

    logger.info("Solving network...")
    assert lower_bound.shape[0] == settings.num_points, \
        f"invalid lower bound shape {lower_bound.shape}, expected ({settings.num_points}, x)"
    assert upper_bound.shape[0] == settings.num_points, \
        f"invalid upper bound shape {upper_bound.shape}, expected ({settings.num_points}, x)"
    assert np_points.shape[0] == settings.num_points, \
        f"invalid points shape {np_points.shape}, expected ({settings.num_points}, x)"

    start = timer()
    (dominant_class, nlb, nub) = eran.analyze_classification_box(Interval(lower_bound, upper_bound))
    end = timer()
    elapsed = end - start
    total_time += elapsed

    certified = dominant_class == label.item()

    if certified:
        logger.info(f"Successfully verified class {dominant_class}")
        verified_same += 1
    elif dominant_class == -1:
        logger.info(f"Failed to verify class {label.item()}")
        not_verified += 1
    else:
        logger.info(f"Wrongly verified class {dominant_class} instead of {label.item()}")
        verified_different += 1

    logger.info(f"Time for this round: {elapsed}s. Total time: {total_time}s.")
    logger.info(f"Tested {iterations} data points out of which {correct_predictions} were correctly predicted.")
    logger.info(f"Successfully verified {verified_same} samples, unable to verify {not_verified} samples. " +
                f"{verified_different} were falsely verified as a different class.")
