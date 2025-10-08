import abc
import numpy as np
import numpy.typing as npt
import typing
from dataclasses import dataclass
from RL_Trading.prj.app.core.classification.services.utils import calculate_matrix_confidence_interval


@dataclass(frozen=True)
class LossFunction(abc.ABC):

    @abc.abstractmethod
    def calculate_cost(self, confusion_matrices: typing.List[npt.NDArray[float]]) -> float:
        pass


@dataclass(frozen=True)
class WeightedCost(LossFunction):
    cost_matrix: npt.NDArray[npt.NDArray[float]]

    def calculate_cost(self, confusion_matrices: typing.List[npt.NDArray[float]]) -> float:
        return np.sum(np.mean(confusion_matrices, axis=0) * self.cost_matrix)


@dataclass(frozen=True)
class OverlapCost(LossFunction):

    def calculate_cost(self, confusion_matrices: typing.List[npt.NDArray[float]]) -> float:
        mean_matrix, margin_of_error = calculate_matrix_confidence_interval(confusion_matrices)
        upper_bound_matrix = mean_matrix + margin_of_error
        lower_bound_matrix = mean_matrix - margin_of_error
        metric_q1_q3 = self._compute_intersection(upper_bound_matrix[0][0], upper_bound_matrix[0][2],
                                                  lower_bound_matrix[0][0], lower_bound_matrix[0][2])
        metric_q1_q7 = self._compute_intersection(upper_bound_matrix[0][0], upper_bound_matrix[2][0],
                                                  lower_bound_matrix[0][0], lower_bound_matrix[2][0])
        metric_q9_q3 = self._compute_intersection(upper_bound_matrix[2][2], upper_bound_matrix[0][2],
                                                  lower_bound_matrix[2][2], lower_bound_matrix[0][2])
        metric_q9_q7 = self._compute_intersection(upper_bound_matrix[2][2], upper_bound_matrix[2][0],
                                                  lower_bound_matrix[2][2], lower_bound_matrix[2][0])
        return metric_q1_q3 + metric_q1_q7 + metric_q9_q3 + metric_q9_q7

    @staticmethod
    def _compute_intersection(
            max_first_interval: float,
            max_second_interval: float,
            min_first_interval: float,
            min_second_interval: float
    ) -> float:
        return min(max_first_interval, max_second_interval) - max(min_first_interval, min_second_interval)
