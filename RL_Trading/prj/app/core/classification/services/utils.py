import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as sts
import typing


def calculate_matrix_confidence_interval(
        matrices: typing.List[npt.NDArray[float]],
        confidence_level: typing.Optional[float] = 0.95
) -> typing.Tuple[npt.NDArray[float], npt.NDArray[float]]:
    mean_matrix = np.mean(matrices, axis=0)
    std_matrix = np.std(matrices, ddof=1, axis=0)
    # Degrees of freedom for the t-distribution
    degrees_of_freedom = len(matrices) - 1
    # Calculate the t-value for the given confidence level and degrees of freedom
    t_value = sts.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    # Calculate the standard error of the mean
    standard_error = std_matrix / np.sqrt(len(matrices))
    # Calculate the margin of error for each element of the matrix using the t-distribution
    margin_of_error = t_value * standard_error
    # Return mean and margin of error matrices
    return mean_matrix, margin_of_error


def calculate_dataframe_confidence_interval(
        df: pd.DataFrame,
        confidence_level: typing.Optional[float] = 0.95
) -> typing.Tuple[npt.NDArray[float], npt.NDArray[float]]:
    mean_values = df.mean()
    # Degrees of freedom for the t-distribution
    degrees_of_freedom = len(df) - 1
    # Calculate the t-value for the given confidence level and degrees of freedom
    t_value = sts.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    # Calculate the standard error of the mean (with 1 degree of freedom for the std computation)
    standard_error = df.sem()
    # Calculate the margin of error using the t-distribution
    margin_of_error = t_value * standard_error
    # Return mean and margin of error
    return mean_values, margin_of_error
