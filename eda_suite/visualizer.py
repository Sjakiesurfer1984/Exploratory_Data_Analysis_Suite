# ==============================================================================
# visualiser.py
#
# Generates all plots for the EDA Suite. Responsible for *visual representation*
# only. It never performs statistical calculations.
#
# Author: Tim Vos
# ==============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Optional
from .schema import SchemaManager
import io


class Visualizer:
    """
    Handles the creation of plots and charts for Exploratory Data Analysis.

    This class encapsulates all plotting logic (Matplotlib, Seaborn) and
    automatically caches figures in memory for report generation.
    """

    def __init__(self, dataframe: pd.DataFrame, schema_manager: SchemaManager):
        self._df = dataframe
        self._schema = schema_manager
        self._plot_cache: List[io.BytesIO] = []

        sns.set_theme(style="whitegrid")

    # ==========================================================================
    # Utility helpers
    # ==========================================================================

    def get_plot_cache(self) -> List[io.BytesIO]:
        """Return all cached plots (used for Word export)."""
        return self._plot_cache

    def _save_plot_to_cache(self) -> None:
        """Save current Matplotlib figure into an in-memory buffer."""
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight")
        img_buffer.seek(0)
        self._plot_cache.append(img_buffer)

    def _format_title(self, dataset_label: Optional[str], base_title: str) -> str:
        """
        Generate a clean, sentence-case, dataset-aware title.
        Converts American → Australian spelling automatically.
        """
        # Sentence case
        title = base_title.strip().capitalize()

        # Add dataset context (e.g. "Yeo-Johnson + scaled")
        if dataset_label:
            title = f"{dataset_label.strip().capitalize()}: {title}"

        # Replace American spellings
        replacements = {
            "Color": "Colour",
            "color": "colour",
            "Normalization": "Normalisation",
            "Normalization": "Normalisation",
            "Center": "Centre",
            "Behavior": "Behaviour",
        }
        for us, au in replacements.items():
            title = title.replace(us, au)

        return title

    # ==========================================================================
    # Core plotting functions
    # ==========================================================================

    def plot_distribution(self, columns: Union[str, List[str]], dataset_label: Optional[str] = None) -> None:
        """Plot distributions for numerical or categorical columns."""
        columns_to_plot = [columns] if isinstance(columns, str) else columns

        for col_name in columns_to_plot:
            orig_col = self._schema.get_original_name(col_name)
            disp_col = self._schema.get_display_name(orig_col)
            plt.figure(figsize=(10, 6))

            if self._df[orig_col].dtype in ["int64", "float64"]:
                sns.histplot(self._df[orig_col], kde=True)
                plt.title(self._format_title(dataset_label, f"distribution of {disp_col}"))
                plt.xlabel(disp_col)
                plt.ylabel("Count")
            else:
                sns.countplot(
                    y=self._df[orig_col],
                    order=self._df[orig_col].value_counts().index,
                )
                plt.title(self._format_title(dataset_label, f"frequency of {disp_col}"))
                plt.xlabel("Frequency")
                plt.ylabel(disp_col)

            self._save_plot_to_cache()
            plt.show()
            plt.close()

    def plot_scatter(self, x_col: str, y_col: str, dataset_label: Optional[str] = None):
        """Plot a scatter chart for two numerical variables."""
        orig_x = self._schema.get_original_name(x_col)
        orig_y = self._schema.get_original_name(y_col)
        disp_x = self._schema.get_display_name(orig_x)
        disp_y = self._schema.get_display_name(orig_y)

        plot_data = self._df[[orig_x, orig_y]].dropna()
        if plot_data.empty:
            print(f"Skipping plot: no overlapping data for '{disp_y}' vs. '{disp_x}'.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=plot_data,
            x=orig_x,
            y=orig_y,
            s=30,
            alpha=0.6,
            color="red",
        )

        plt.title(self._format_title(dataset_label, f"{disp_y} vs {disp_x}"))
        plt.xlabel(disp_x)
        plt.ylabel(disp_y)

        self._save_plot_to_cache()
        plt.show()
        plt.close()

    def plot_boxplots(
        self,
        numerical_cols: Union[str, List[str]],
        group_by_col: Optional[str] = None,
        dataset_label: Optional[str] = None,
    ) -> None:
        """Plot single or grouped boxplots."""
        cols = [numerical_cols] if isinstance(numerical_cols, str) else numerical_cols
        orig_cols = [self._schema.get_original_name(c) for c in cols]

        # Grouped case
        if group_by_col:
            group_orig = self._schema.get_original_name(group_by_col)
            group_disp = self._schema.get_display_name(group_orig)
            for orig_col, disp_col in zip(orig_cols, cols):
                plt.figure(figsize=(12, 7))
                sns.boxplot(data=self._df, x=group_orig, y=orig_col)
                plt.title(
                    self._format_title(dataset_label, f"distribution of {disp_col} by {group_disp}")
                )
                plt.xlabel(group_disp)
                plt.ylabel(disp_col)
                plt.xticks(rotation=45)
                self._save_plot_to_cache()
                plt.show()
                plt.close()
        else:
            # Multiple numerical columns
            plt.figure(figsize=(12, 7))
            df_melted = self._df[orig_cols].melt(var_name="Variable", value_name="Value")
            sns.boxplot(data=df_melted, x="Variable", y="Value")
            plt.title(self._format_title(dataset_label, "box plots of selected columns"))
            plt.xticks(rotation=45)
            self._save_plot_to_cache()
            plt.show()
            plt.close()

    def plot_grouped_boxgrid(
    self,
    numerical_cols: list[str],
    group_col: str,
    dataset_label: str | None = None,
    ) -> None:
        """
        Plot a grid of boxplots showing multiple numerical columns grouped by a categorical variable.
    
        Args:
            numerical_cols (list[str]): Numerical columns to plot.
            group_col (str): Categorical column to group by.
            dataset_label (str | None): Optional label (e.g. 'Raw data') for the title.
        """
        if not numerical_cols:
            print("No numerical columns provided for grouped boxplot grid.")
            return
    
        df_long = self._df.melt(
            id_vars=group_col, 
            value_vars=numerical_cols, 
            var_name="Feature", 
            value_name="Value"
        )
    
        title_prefix = f"{dataset_label}: " if dataset_label else ""
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df_long, x="Feature", y="Value", hue=group_col)
        plt.title(f"{title_prefix}Distributions of features by {group_col.lower()}")
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.legend(title=group_col)
        plt.tight_layout()
        self._save_plot_to_cache()
        plt.show()

    def plot_pairplot(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        dataset_label: Optional[str] = None,
    ) -> None:
        """Create a pair plot (scatter-plot matrix)."""
        if columns is None:
            columns = self._df.select_dtypes(include=["number"]).columns
        g = sns.pairplot(self._df[columns], hue=hue, diag_kind="kde", corner=True)
        g.fig.suptitle(self._format_title(dataset_label, "pair plot of selected features"), y=1.02)
        plt.show()

    def plot_correlation_matrix(
        self,
        stats_calculator,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        dataset_label: Optional[str] = None,
    ) -> None:
        """Plot correlation matrix with automatic sentence-case title."""
        corr = stats_calculator.get_correlation_matrix(columns=columns, method=method)
        if corr.empty:
            print("No numerical columns for correlation plot.")
            return

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title(self._format_title(dataset_label, f"{method} correlation matrix"))
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_plot_to_cache()
        plt.show()
        plt.close()

    def plot_covariance_heatmap(
        self,
        cov_matrix: pd.DataFrame,
        dataset_label: Optional[str] = None,
    ) -> None:
        """Plot covariance matrix with adaptive numeric formatting."""
        if cov_matrix.empty:
            print("Covariance matrix empty. Skipping plot.")
            return

        max_val = np.abs(cov_matrix.values).max()
        if max_val >= 1e6:
            fmt = ".2e"
        elif max_val >= 1:
            fmt = ".2f"
        elif max_val >= 1e-3:
            fmt = ".3f"
        else:
            fmt = ".2e"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cov_matrix,
            annot=True,
            fmt=fmt,
            cmap="coolwarm",
            square=True,
            cbar_kws={"label": "Covariance"},
        )
        plt.title(self._format_title(dataset_label, "covariance matrix"))
        plt.tight_layout()
        self._save_plot_to_cache()
        plt.show()
        plt.close()

    def plot_pca_scree(
        self,
        pca_df: pd.DataFrame,
        dataset_label: str | None = None
    ) -> None:
        """
        Plot a PCA scree plot showing explained variance per component.

        Args:
            pca_df (pd.DataFrame): DataFrame returned by get_pca_components().
            dataset_label (str | None): Optional dataset label for the title.
        """
        if pca_df.empty:
            print("No PCA data to plot.")
            return

        title_prefix = f"{dataset_label}: " if dataset_label else ""
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x="Component",
            y="ExplainedVarianceRatio",
            data=pca_df,
            marker="o"
        )
        sns.scatterplot(
            x="Component",
            y="ExplainedVarianceRatio",
            data=pca_df,
            color="red",
            s=80,
            legend=False
        )
        plt.plot(
            pca_df["Component"],
            pca_df["CumulativeVariance"],
            linestyle="--",
            color="grey",
            alpha=0.7,
            label="Cumulative variance"
        )
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance ratio")
        plt.title(f"{title_prefix}PCA scree plot")
        plt.legend()
        plt.tight_layout()
        self._save_plot_to_cache()
        plt.show()
