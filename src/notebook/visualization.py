import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df: pd.DataFrame, hue_col: str) -> sns.axisgrid.PairGrid:
    """Creates a pairplot for each numerical column in the DataFrame,
    with the specified column as the hue."""
    non_boolean_columns = df.select_dtypes(exclude=["bool"]).columns
    columns_to_plot = non_boolean_columns.tolist() + [hue_col]
    return sns.pairplot(df[columns_to_plot], hue=hue_col)

def plot_boxplots_by_category(df: pd.DataFrame, x_col: str):
    """Creates a series of boxplots for each numerical column in the DataFrame,
    categorized by the specified column."""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numerical_columns)
    plt.figure(figsize=(n_cols * 5, 4))
    for i, col in enumerate(numerical_columns):
        plt.subplot(1, n_cols, i + 1)
        sns.boxplot(x=x_col, y=col, data=df)
    plt.tight_layout()
    return plt.show()

def plot_categorical_distributions(df: pd.DataFrame, hue_col: str, rotated_xticks_cols: list):
    """Creates a series of countplots for each categorical column in the DataFrame."""
    categorical_columns = df.select_dtypes(include=['bool', 'category', 'object']).columns
    categorical_columns = categorical_columns.drop(hue_col, errors='ignore')

    # Determine the number of rows and columns for subplots
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols  # Calculate rows needed

    # Create a figure with subplots
    plt.figure(figsize=(n_cols * 5, n_rows * 4))

    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=col, hue=hue_col, data=df)
        plt.title(f'Distribution of {col} by {hue_col}')
        if col in rotated_xticks_cols: plt.xticks(rotation=45)  # Rotate the x-axis labels
        plt.tight_layout()

    plt.show()


def plot_sorted_correlation(df: pd.DataFrame, target_col: str):
    """Creates a heatmap of the correlation matrix,
    sorted by the absolute value of the correlation with the target column."""
    df_encoded = pd.get_dummies(df)
    corr = df_encoded.corr()

    sorted_indices = corr.abs().sort_values(by=target_col, ascending=False).index
    sorted_corr = corr.loc[sorted_indices, sorted_indices]

    plt.figure(figsize=(12, 10))

    sns.heatmap(sorted_corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, annot_kws={'size': 8})

    # Rotate x-axis labels and set color
    plt.xticks(rotation=45, ha='right', color='black')
    plt.yticks(color='black')

    # Highlight the target_col column and row
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if label.get_text() == target_col:
            label.set_color('red')
    for i, label in enumerate(plt.gca().get_yticklabels()):
        if label.get_text() == target_col:
            label.set_color('red')

    plt.title(f'Correlation Matrix Ordered by Absolute {target_col} Correlation')
    plt.show()
