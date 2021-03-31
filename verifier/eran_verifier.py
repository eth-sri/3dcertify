import sys
from typing import List

import numpy as np

from relaxations.interval import Interval
from relaxations.linear_bounds import LinearBounds

sys.path.insert(0, './ERAN/ELINA/python_interface/')
sys.path.insert(0, './ERAN/tf_verify/')

from eran import ERAN
from analyzer import layers, Analyzer
from elina_abstract0 import elina_abstract0_free


class EranResources:
    """Simple data object holding resources used by the ERAN verifier"""
    expr_size: int

    lower_bound: np.ndarray
    upper_bound: np.ndarray

    lexpr_weights: np.ndarray
    uexpr_weights: np.ndarray

    lexpr_cst: np.ndarray
    uexpr_cst: np.ndarray

    lexpr_dim: np.ndarray
    uexpr_dim: np.ndarray


class EranVerifier:
    """Wrapper object for the ERAN verifier to have a clean interface with proper types."""
    __TIMEOUT_LP = 1_000_000
    __TIMEOUT_MILP = 1_000_000

    def __init__(self, model):
        self.eran = ERAN(model=model, is_onnx=True)

    def analyze_classification_box(self, bounds: Interval):
        (dominant_class, _, nlb, nub, _) = self.eran.analyze_box(
            specLB=bounds.lower_bound,
            specUB=bounds.upper_bound,
            domain='deeppoly',
            timeout_lp=EranVerifier.__TIMEOUT_LP,
            timeout_milp=EranVerifier.__TIMEOUT_MILP,
            use_default_heuristic=True,
            testing=True
        )
        return dominant_class, nlb, nub

    def analyze_classification_linear(self, bounds: Interval, constraints: LinearBounds, params: List[Interval]):
        res = self.__eran_resources(bounds, constraints, params)

        (dominant_class, _, nlb, nub, _) = self.eran.analyze_box(
            specLB=res.lower_bound,
            specUB=res.upper_bound,
            domain='deeppoly',
            timeout_lp=EranVerifier.__TIMEOUT_LP,
            timeout_milp=EranVerifier.__TIMEOUT_MILP,
            lexpr_weights=res.lexpr_weights,
            lexpr_cst=res.lexpr_cst,
            lexpr_dim=res.lexpr_dim,
            uexpr_weights=res.uexpr_weights,
            uexpr_cst=res.uexpr_cst,
            uexpr_dim=res.uexpr_dim,
            expr_size=res.expr_size,
            use_default_heuristic=True,
            testing=True
        )
        return dominant_class, nlb, nub

    def analyze_segmentation_box(self, bounds: Interval, correct_labels: np.ndarray, valid_labels: np.ndarray,
                                 num_total_classes: int):
        (dominant_classes, nlb, nub) = EranVerifier.__analyze_segmentation_eran(
            eran=self.eran,
            true_labels=correct_labels,
            valid_classes=valid_labels,
            num_total_classes=num_total_classes,
            specLB=bounds.lower_bound,
            specUB=bounds.upper_bound,
            domain='deeppoly',
            timeout_lp=EranVerifier.__TIMEOUT_LP,
            timeout_milp=EranVerifier.__TIMEOUT_MILP,
            use_default_heuristic=True,
            testing=True
        )
        return dominant_classes, nlb, nub

    def analyze_segmentation_linear(self, bounds: Interval, constraints: LinearBounds, params: List[Interval],
                                    correct_labels: np.ndarray, valid_labels: np.ndarray, num_total_classes: int):
        res = self.__eran_resources(bounds, constraints, params)

        (dominant_classes, nlb, nub) = EranVerifier.__analyze_segmentation_eran(
            eran=self.eran,
            true_labels=correct_labels,
            valid_classes=valid_labels,
            num_total_classes=num_total_classes,
            specLB=res.lower_bound,
            specUB=res.upper_bound,
            domain='deeppoly',
            timeout_lp=EranVerifier.__TIMEOUT_LP,
            timeout_milp=EranVerifier.__TIMEOUT_MILP,
            lexpr_weights=res.lexpr_weights,
            lexpr_cst=res.lexpr_cst,
            lexpr_dim=res.lexpr_dim,
            uexpr_weights=res.uexpr_weights,
            uexpr_cst=res.uexpr_cst,
            uexpr_dim=res.uexpr_dim,
            expr_size=res.expr_size,
            use_default_heuristic=True,
            testing=True
        )
        return dominant_classes, nlb, nub

    # adapted from ERAN/tf_verify/ERAN to support segmentation
    @staticmethod
    def __analyze_segmentation_eran(eran: ERAN, true_labels, valid_classes, num_total_classes, specLB, specUB, domain,
                                    timeout_lp, timeout_milp, use_default_heuristic,
                                    lexpr_weights=None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None,
                                    uexpr_cst=None, uexpr_dim=None, expr_size=0, testing=False, prop=-1,
                                    spatial_constraints=None):
        assert domain == "deeppoly", "domain isn't valid, must be 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        eran.nn = layers()
        eran.nn.specLB = specLB
        eran.nn.specUB = specUB

        execute_list, output_info = eran.optimizer.get_deeppoly(eran.nn, specLB, specUB, lexpr_weights, lexpr_cst,
                                                                lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim,
                                                                expr_size, spatial_constraints)
        analyzer = Analyzer(execute_list, eran.nn, domain, timeout_lp, timeout_milp, None, use_default_heuristic, -1, prop, testing)
        dominant_classes, nlb, nub = EranVerifier.__analyze_segmentation_analyzer(analyzer, true_labels, valid_classes, num_total_classes)
        return dominant_classes, nlb, nub

    # adapted from ERAN/tf_verify/Analyzer to support segmentation
    @staticmethod
    def __analyze_segmentation_analyzer(analyzer: Analyzer, true_labels, valid_classes, num_classes: int):

        element, nlb, nub = analyzer.get_abstract0()

        number_points = len(true_labels)
        dominant_classes = []

        for n in range(number_points):
            dominant_class = -1
            label_failed = []
            candidate_labels = valid_classes
            true_label = true_labels[n]

            certified = True
            for candidate_label in candidate_labels:
                point_offset = n * num_classes
                if candidate_label != true_label and not analyzer.is_greater(analyzer.man, element, point_offset + true_label, point_offset + candidate_label, analyzer.use_default_heuristic):
                    certified = False
                    label_failed.append(candidate_label)
                    break
            if certified:
                dominant_class = true_label
            dominant_classes.append(dominant_class)

        elina_abstract0_free(analyzer.man, element)
        return dominant_classes, nlb, nub

    @staticmethod
    def __eran_resources(bounds: Interval, constraints: LinearBounds, params: List[Interval]) -> EranResources:
        res = EranResources()

        res.lower_bound = np.append(bounds.lower_bound, [p.lower_bound for p in params])
        res.upper_bound = np.append(bounds.upper_bound, [p.upper_bound for p in params])

        expr_size = len(params)
        res.expr_size = expr_size
        num_indices = len(res.lower_bound.flatten())
        param_indices = np.arange(num_indices - expr_size, num_indices)

        res.lexpr_weights = np.append(constraints.lower_slope.flatten(), np.zeros(expr_size * expr_size))
        res.uexpr_weights = np.append(constraints.upper_slope.flatten(), np.zeros(expr_size * expr_size))

        res.lexpr_cst = np.append(constraints.lower_offset.flatten(), [p.lower_bound for p in params])
        res.uexpr_cst = np.append(constraints.upper_offset.flatten(), [p.upper_bound for p in params])

        res.lexpr_dim = np.repeat([param_indices], num_indices, axis=0).flatten()
        res.uexpr_dim = np.repeat([param_indices], num_indices, axis=0).flatten()

        assert len(res.lower_bound) == num_indices
        assert len(res.upper_bound) == num_indices
        assert len(res.lexpr_weights) == num_indices * expr_size
        assert len(res.uexpr_weights) == num_indices * expr_size
        assert len(res.lexpr_cst) == num_indices
        assert len(res.uexpr_cst) == num_indices
        assert len(res.lexpr_dim) == num_indices * expr_size
        assert len(res.uexpr_dim) == num_indices * expr_size
        return res
