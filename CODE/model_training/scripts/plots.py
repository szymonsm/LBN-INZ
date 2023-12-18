import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scripts.essentials import *

def plot_lagged_correlations(df, target):
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


def close_price_statistics_by_year(df1, date_col, close_col, start_date=None, end_date=None):
    """
    Extract statistics for the 'Close' column over different time intervals, grouped by year.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - date_col (str): Column name for the date.
    - close_col (str): Column name for the 'Close' prices.
    - start_date (str or None): Start date for statistics (format: 'YYYY-MM-DD').
    - end_date (str or None): End date for statistics (format: 'YYYY-MM-DD').

    Returns:
    - pd.DataFrame: DataFrame with statistics for the 'Close' column, each row representing a year.
    """
    # Ensure the DataFrame is sorted by date
    df = df1.copy()
    df.sort_values(by=date_col, inplace=True)

    # Filter data based on the specified date range
    if start_date is not None and end_date is not None:
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    # Extract date and close columns
    date_series = df[date_col]
    close_series = df[close_col]

    # Extract year from the date
    df['Year'] = pd.to_datetime(df[date_col]).dt.year

    # Calculate mean absolute difference for different time intervals
    one_day_diff = close_series.diff(1).abs()
    one_week_diff = close_series.diff(5).abs()
    two_weeks_diff = close_series.diff(10).abs()
    one_month_diff = close_series.diff(20).abs()
    two_months_diff = close_series.diff(40).abs()

    # Group by year and calculate statistics
    grouped_stats = df.groupby('Year')[close_col].agg([
        ('Min', 'min'),
        ('Max', 'max'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Percentile_25', lambda x: np.percentile(x, 25)),
        ('Percentile_50', lambda x: np.percentile(x, 50)),
        ('Percentile_75', lambda x: np.percentile(x, 75)),
        ('Mean_Abs_Diff_1D', lambda x: one_day_diff[x.index].mean()),
        ('Mean_Abs_Diff_1W', lambda x: one_week_diff[x.index].mean()),
        ('Mean_Abs_Diff_2W', lambda x: two_weeks_diff[x.index].mean()),
        ('Mean_Abs_Diff_1M', lambda x: one_month_diff[x.index].mean()),
        ('Mean_Abs_Diff_2M', lambda x: two_months_diff[x.index].mean())
    ]).reset_index()

    return grouped_stats

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

def plot_actual_vs_predicted(y_actual, y_pred, y_base, idx, title='Actual vs Predicted', xlabel='Index', ylabel='Values'):
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
    # put idx over the marker of y_pred
    for i, id in enumerate(idx):
        plt.annotate(id, (i, y_pred[i]))
    plt.plot(y_base, label='Baseline', linestyle='--', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()