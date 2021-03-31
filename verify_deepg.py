import argparse

import numpy as np
import onnx
import onnxruntime
import torch
from tqdm import tqdm

from data_processing import datasets
from pointnet.model import PointNet
from relaxations.deepg_bounds import load_spec
from util import onnx_converter
from util.argparse import absolute_path
from util.experiment import Experiment
from util.math import DEFAULT_SEED
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

eran = EranVerifier(model=onnx_model)
test_samples = len(test_data)
interval = len(test_data) // 100

correct_predictions = 0
verified_same = 0
verified_different = 0
not_verified = 0
iterations = 0

timer = Timer()

for counter, i in enumerate(range(0, test_samples, interval)):

    iterations += 1
    np_points, faces, label = test_data[i]
    points = torch.from_numpy(np_points)
    points = torch.unsqueeze(points, dim=0)

    assert np_points.shape[0] == settings.num_points, \
        f"invalid points shape {np_points.shape}, expected ({settings.num_points}, x)"

    prediction = torch_model(points.transpose(2, 1))
    max_prediction = prediction.data.max(1)[1].item()
    correct = max_prediction == label

    if not correct:
        logger.info(
            "Incorrect prediction, skipping. True label was {}, prediction was {}".format(label, max_prediction))
        continue

    correct_predictions += 1

    session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    input_data = np.expand_dims(np.transpose(np_points.copy()), axis=(0, -1))
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)

    np.testing.assert_allclose(prediction.detach().numpy(), outputs[0], rtol=1e-2, atol=1e-3)

    deepg_spec = load_spec(settings.spec_dir, counter)

    checkpoints_sample = checkpoints[str(i)] if str(i) in checkpoints else {}

    certified = True
    timer.start()

    progress_bar = tqdm(deepg_spec, desc=f"Object {counter}", unit="interval")
    for params, bounds, constraints in progress_bar:

        interval_key = 'x'.join([f"[{i.lower_bound:.4f},{i.upper_bound:.4f}]" for i in params])
        if interval_key in checkpoints_sample:
            interval_certified = checkpoints_sample[interval_key]
        else:
            (dominant_class, nlb, nub) = eran.analyze_classification_linear(bounds, constraints, params)

            interval_certified = dominant_class == label.item()
            checkpoints_sample[interval_key] = interval_certified

        if not interval_certified:
            certified = False

    elapsed = timer.stop()

    checkpoints[str(i)] = checkpoints_sample
    experiment.store_checkpoints(checkpoints)

    if certified:
        logger.info(f"Successfully certify class {label.item()} in {len(deepg_spec)} intervals")
        verified_same += 1
    else:
        logger.info(f"Failed to certify class {label.item()}")

    logger.info(f"Time for this round: {elapsed}s. Total time: {timer.get()}s.")
    logger.info(f"Tested {iterations} data points out of which {correct_predictions} were correctly predicted.")
    logger.info(f"Successfully certified {verified_same} samples.")
