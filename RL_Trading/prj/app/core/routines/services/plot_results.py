import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as sts
import typing
from RL_Trading.prj.app.core.classification.services.utils import calculate_matrix_confidence_interval
from RL_Trading.prj.app.core.classification.services.utils import calculate_dataframe_confidence_interval


class Plotter(object):

    def __new__(cls, *args, **kwargs):
        raise NotImplemented("This class cannot be constructed. Use static method in order to initialize it.")

    def __init__(self):
        raise NotImplemented("This class cannot be constructed. Use static method in order to initialize it.")

    @staticmethod
    def plot_confusion_matrix(
            confusion_matrices: typing.List[npt.NDArray[float]],
            ax: matplotlib.axes.SubplotBase,
            title: str
    ):
        means, margins = calculate_matrix_confidence_interval(confusion_matrices)
        # Plot training confusion matrix
        ax.imshow(means, interpolation='nearest', cmap='viridis')
        num_classes = len(means)
        for i in range(num_classes):
            for j in range(num_classes):
                lower_interval = means[i, j] - margins[i, j]
                upper_interval = means[i, j] + margins[i, j]
                ax.text(j, i, f'({lower_interval:.3f}, {upper_interval:.3f})', ha='center', va='center')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_title(title)

    @staticmethod
    def plot_confusion_matrix_corners(
            confusion_matrices_train: typing.List[npt.NDArray[float]],
            confusion_matrices_valid: typing.List[npt.NDArray[float]],
            ax: matplotlib.axes.SubplotBase,
            title: str
    ):
        means_train, margins_train = calculate_matrix_confidence_interval(confusion_matrices_train)
        means_valid, margins_valid = calculate_matrix_confidence_interval(confusion_matrices_valid)
        # Keep corners of the matrices
        values_train = means_train[[0, 2]][:, [0, 2]].T.flatten()
        errors_train = margins_train[[0, 2]][:, [0, 2]].T.flatten()
        values_valid = means_valid[[0, 2]][:, [0, 2]].T.flatten()
        errors_valid = margins_valid[[0, 2]][:, [0, 2]].T.flatten()
        # Plot error bars
        ax.errorbar(np.arange(0.0, 4.0, 1.0), values_train, yerr=errors_train, fmt='o', color='blue', label='train')
        ax.errorbar(np.arange(0.1, 4.1, 1.0), values_valid, yerr=errors_valid, fmt='o', color='red', label='validation')
        # Format plot
        ax.set_title(title)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_xticks(np.arange(0.05, 4.05, 1.0), labels=['(0, 0)', '(0, 2)', '(2, 0)', '(2, 2)'])
        ax.grid(alpha=0.5, linestyle=':')

    @staticmethod
    def plot_feature_importances(
            feature_importances_df: pd.DataFrame,
            ax: matplotlib.axes.SubplotBase,
            title: str
    ):
        values, errors = calculate_dataframe_confidence_interval(feature_importances_df)
        ax.errorbar(feature_importances_df.columns, values, yerr=errors, fmt='o', color='blue')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90)
        ax.grid(alpha=0.5, linestyle=':')


