import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


def create_pairplot_for_numerical_columns(df: pd.DataFrame, hue_col: str) -> tuple[Figure, Axes]:
    """Creates a pairplot for each numerical column in the DataFrame,
    with the specified column as the hue."""
    non_boolean_columns = df.select_dtypes(exclude=["bool"]).columns
    columns_to_plot = non_boolean_columns.tolist() + [hue_col]
    pair_grid = sns.pairplot(df[columns_to_plot], hue=hue_col)
    return pair_grid.figure, pair_grid.axes


def create_boxplots_for_numerical_columns_by_category(df: pd.DataFrame, x_col: str) -> Figure:
    """Creates a series of boxplots for each numerical column in the DataFrame,
    categorized by the specified column."""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numerical_columns)
    fig = plt.figure(figsize=(n_cols * 5, 4))
    for i, col in enumerate(numerical_columns):
        plt.subplot(1, n_cols, i + 1)
        sns.boxplot(x=x_col, y=col, data=df)
    plt.tight_layout()
    return fig


def create_categorical_countplots(
    df: pd.DataFrame, hue_col: str, rotated_xticks_cols: list
) -> Figure:
    """Creates a series of countplots for each categorical column in the DataFrame."""
    categorical_columns = df.select_dtypes(
        include=["bool", "category", "object"]
    ).columns
    categorical_columns = categorical_columns.drop(hue_col, errors="ignore")

    # Determine the number of rows and columns for subplots
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols  # Calculate rows needed

    # Create a figure with subplots
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4))

    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=col, hue=hue_col, data=df)
        plt.title(f"Distribution of {col} by {hue_col}")
        if col in rotated_xticks_cols:
            plt.xticks(rotation=45)  # Rotate the x-axis labels
        plt.tight_layout()

    return fig


def create_target_sorted_correlation_heatmap(
    df: pd.DataFrame, target_col: str
) -> tuple[Figure, Axes]:
    """Creates a heatmap of the correlation matrix, sorted by the absolute value of the
    correlation with the target column, and returns the figure and axes."""
    df_encoded = pd.get_dummies(df)
    corr = df_encoded.corr()

    sorted_indices = corr.abs().sort_values(by=target_col, ascending=False).index
    sorted_corr = corr.loc[sorted_indices, sorted_indices]

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap on the axes
    ax = sns.heatmap(
        sorted_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        annot_kws={"size": 8},
        ax=ax,
    )

    # Rotate x-axis labels and set color
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="black")
    ax.set_yticklabels(ax.get_yticklabels(), color="black")

    # Highlight the target_col column and row
    for label in ax.get_xticklabels():
        if label.get_text() == target_col:
            label.set_color("red")
    for label in ax.get_yticklabels():
        if label.get_text() == target_col:
            label.set_color("red")

    ax.set_title(f"Correlation Matrix Ordered by Absolute {target_col} Correlation")

    return fig, ax
