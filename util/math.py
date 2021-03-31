import random

import numpy as np
import torch

DEFAULT_SEED = 1823453073


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_point_iou(prediction, actual):
    sum_iou = 0.0
    classes = np.unique(actual)
    for cl in classes:
        cl_prediction = prediction == cl
        cl_actual = actual == cl
        intersection = np.logical_and(cl_prediction, cl_actual).sum()
        union = np.logical_or(cl_prediction, cl_actual).sum()
        if union == 0:
            iou = 1
        else:
            iou = intersection / union
        sum_iou += iou
    return sum_iou / len(classes)


def logits_to_category(logits, expected):
    logits = logits.copy()
    valid_categories = np.unique(expected)
    invalid_categories = list(set(np.arange(logits.shape[0])).difference(set(valid_categories)))
    logits[invalid_categories, :] = np.min(logits) - 1000  # manually deactivate irrelevant logits
    prediction = np.argmax(logits, axis=0)
    return prediction
