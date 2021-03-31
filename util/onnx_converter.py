import onnx
import torch.nn as nn
import torch.onnx

from util import translate_onnx


def __export(model: nn.Module, num_points: int, file):
    dummy_batch_size = 1
    input_features = 3
    dummy_input = torch.randn(dummy_batch_size, input_features, num_points)
    torch.onnx.export(model, (dummy_input,), file, verbose=False)


def convert(model: nn.Module, num_points: int, file):
    __export(model, num_points, file)
    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)
    translate_onnx.translate(onnx_model, remove_first_layer=False)
    return onnx_model
