'''
Plik zawiera funkcje do różnorodnych wykresów, które są wykorzystywane w notebookach.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scripts.essentials import *

def plot_lagged_correlations(df, target):
    """
    Plot the correlations between the target and lagged versions of itself.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The target column.

    Returns:
    - None (Displays the plot).
    """
    # Calculate correlations for all columns and target
    correlations = {}
    for col in df.columns:
        if col != target:
            correlations[col] = calculate_lagged_correlation(col, target, df)

    # Plot correlations for each column
    for col, corr in correlations.items():
        corr.plot(x='Lag', y='Correlation', kind='scatter')
        plt.title(f'Correlation between {col} and {target}')
        plt.show()

def check_and_plot_correlation(df, target_col,method_='spearman'):
    # Calculate the correlation matrix
    corr_matrix = df.corr(method= method_)

    # Get the correlation values for the target column
    target_corr = corr_matrix[target_col]

    # Remove the correlation value of the target column with itself
    target_corr = target_corr.drop(target_col)

    # Sort the correlation values in descending order
    target_corr = target_corr.sort_values(ascending=False)

    # Plot the correlation values
    plt.figure(figsize=(10, 6))
    target_corr.plot(kind='bar')
    plt.title(f'Correlation with {target_col}')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.show()

def plot_series(target_col, cols_to_plot, date_col, data_org,start_date=None, end_date=None):
    """
    Plot time series data in an academic style with the ability to zoom multiple plots under each other.

    Parameters:
    - target_col (str): The target column to be plotted separately.
    - cols_to_plot (list): List of columns to be plotted along with the target.
    - date_col (str): Column name for the date.
    - data (pd.DataFrame): The input DataFrame containing the time series data.

    Returns:
    - None (Displays the plot)
    """
    # Set the date column as the index
    data = data_org.copy()
    data.set_index(date_col, inplace=True)
    if start_date is not None and end_date is not None:
        data = data.loc[start_date:end_date]
    # Plotting target separately
    plt.figure(figsize=(12, 38))
    plt.subplot(len(cols_to_plot) + 1, 1, 1)
    plt.plot(data[target_col], label=target_col, color='blue')
    plt.title(f'Target: {target_col}')
    plt.legend()
    
    # Plotting other columns
    for i, col in enumerate(cols_to_plot):
        plt.subplot(len(cols_to_plot) + 1, 1, i + 2)
        plt.plot(data[col], label=col, color='orange')
        plt.title(f'{col} along with {target_col}')
        plt.legend()

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()

def plot_feature_importance(shap_values, feature_names,window_size):
    """
    Plot the importance of each feature at each step in the window.

    :param shap_values: A 3D array of SHAP values with shape (samples, window_size, num_features).
    :param feature_names: List of feature names.
    """
    # Aggregate SHAP values across all samples
    aggregated_shap = np.mean(np.abs(shap_values[0]), axis=0)

    # Create a plot for each time step in the window
    window_size = aggregated_shap.shape[0]
    num_features = aggregated_shap.shape[1]

    # Set up the plot
    fig, axs = plt.subplots(window_size, 1, figsize=(10, window_size * 2))
    fig.suptitle('Feature Importance in Each Window Step')

    # Plot each time step
    for i in range(window_size):
        axs[i].bar(feature_names, aggregated_shap[i, :])
        axs[i].set_title(f'Time Step {i+1}')
        axs[i].set_ylabel('SHAP Value')
        axs[i].set_xlabel('Features')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_mean_feature_importance(shap_values, feature_names,max_=False):
    """
    Plot the mean importance of each feature across all time steps.

    :param shap_values: A 3D array of SHAP values with shape (samples, window_size, num_features).
    :param feature_names: List of feature names.
    """
    if max_:
    # Aggregate SHAP values across all samples and time steps
      aggregated_shap = np.max(np.abs(shap_values), axis=(0))
    else:
      aggregated_shap = np.mean(np.abs(shap_values), axis=(0))


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, aggregated_shap)
    plt.title('Mean Feature Importance Across All Time Steps')
    plt.ylabel('Mean SHAP Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_density(vector, title='Density Plot', xlabel='Values'):
    """
    Plot the density of a vector.

    Parameters:
    - vector (list, array, or pandas Series): The vector to be plotted.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.

    Returns:
    - None (Displays the plot).
    """
    # Create a density plot using seaborn
    sns.histplot(vector, kde=True)

    # Set plot title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')

    # Show the plot
    plt.show()

def plot_actual_vs_predicted(y_actual, y_pred, y_base, title='Actual vs Predicted', xlabel='Index', ylabel='Values'):
    """
    Plot the actual vs predicted values.

    Parameters:
    - y_actual (array-like): Actual values.
    - y_pred (array-like): Predicted values.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None (Displays the plot).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', linestyle='--', marker='o')
    plt.plot(y_base, label='Baseline', linestyle='--', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()