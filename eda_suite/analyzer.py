# ==============================================================================
# analyzer.py
#
# Provides the EDAAnalyzer class — the single high-level interface (Facade)
# for the EDA Suite.  It orchestrates all low-level components such as
# the Profiler, StatisticsCalculator, Visualiser, SchemaManager, Cleaner,
# and ReportGenerator.
#
# Author: Tim Vos
# ==============================================================================

from __future__ import annotations
from typing import Dict, Union, List, Any, Optional
import numpy as np
import uuid
import json
import os
from datetime import datetime
import io

# --- Component imports for type hints ----------------------------------------
from .profiler import DataProfiler
from .statistics import StatisticsCalculator
from .visualizer import Visualizer
from .schema import SchemaManager
from .cleaner import DataCleaner
from .report_generator import ReportGenerator


class EDAAnalyzer:
    """
    High-level facade that orchestrates the entire Exploratory Data Analysis process.

    This class does not perform analysis itself; it delegates all specialised
    work to injected components while exposing a simple, cohesive API to users.
    """

    def __init__(
        self,
        profiler: DataProfiler,
        stats: StatisticsCalculator,
        visualizer: Visualizer,
        schema: SchemaManager,
        cleaner: DataCleaner,
        report_generator: ReportGenerator,
        name: str = "analyzer"
    ) -> None:
        """
        Initialise the EDAAnalyzer and store injected dependencies.

        Args:
            profiler (DataProfiler): Performs data-type heuristics and profiling.
            stats (StatisticsCalculator): Handles statistical computations.
            visualizer (Visualizer): Generates plots and caches them for reports.
            schema (SchemaManager): Manages display-name mappings for columns.
            cleaner (DataCleaner): Provides data-cleaning utilities.
            report_generator (ReportGenerator): Creates formatted output documents.
        """
        self._profiler = profiler
        self._stats = stats
        self._visualizer = visualizer
        self._schema = schema
        self._cleaner = cleaner
        self._report_generator = report_generator
        self._type_overrides: Dict[str, str] = {}
        
        # ----------------------------------------------------------------------
        # Instance identification
        # ----------------------------------------------------------------------
        self.name: str = name  # used for file naming and session tracking

        # ----------------------------------------------------------------------
        # Version-tracking metadata
        # ----------------------------------------------------------------------
        self.session_id: str = uuid.uuid4().hex[:8]
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            from eda_suite import __version__ as suite_version
        except ImportError:
            suite_version = "unknown"
        self.suite_version: str = suite_version

        safe_timestamp = self.created_at.replace(":", "-").replace(" ", "_")
        self.version_tag: str = f"EDA_{safe_timestamp}_{self.session_id}"

        self._log_dir: str = os.path.join(os.getcwd(), ".logs")
        os.makedirs(self._log_dir, exist_ok=True)
        self._save_session_metadata()

    # ==========================================================================
    # PROFILER METHODS
    # ==========================================================================
    def get_dataframe(self) -> pd.DataFrame:
        return self._profiler.get_dataframe()
        
    def set_column_type(self, column: str, new_type: str) -> None:
        """
        Override the automatically inferred column type.

        Args:
            column (str): Column name to override.
            new_type (str): One of {'numerical', 'categorical', 'id'}.
        """
        valid = ["numerical", "categorical", "id"]
        if new_type.lower() not in valid:
            print(f"Error: Invalid type '{new_type}'. Must be one of {valid}")
            return
        self._type_overrides[column] = new_type.lower()
        print(f"Override set: '{column}' → {new_type.lower()}.\n")

    def show_profile(self) -> None:
        """Print a concise heuristic summary of the dataset."""
        profile = self._profiler.get_summary(overrides=self._type_overrides)
        print("--- DataFrame Heuristic Profile ---")
        print(f"Shape: {profile['shape'][0]} × {profile['shape'][1]}")
        print("\nNumerical Columns:", profile["numerical_columns (heuristic)"])
        print("Categorical Columns:", profile["categorical_columns (heuristic)"])
        print("ID Columns:", profile["id_columns (heuristic)"])
        print("-----------------------------------\n")

    def show_mixed_type_report(self) -> None:
        """List columns containing multiple data types."""
        report = self._profiler.get_mixed_type_report()
        print("--- Mixed Data Type Report ---")
        if not report:
            print("No columns with mixed types.")
        else:
            for col, types in report.items():
                print(f"• {col}: {types}")
        print("------------------------------\n")

    def show_missing_values(self) -> None:
        """Display counts of missing (NaN) values per column."""
        missing_df = self._profiler.get_missing_values()
        print("--- Missing Values Report ---")
        if missing_df.empty:
            print("No missing values found.")
        else:
            print(missing_df.to_string())
        print("---------------------------\n")

    def show_value_occurrences(
        self,
        values_to_find: Union[Any, list[Any]],
        columns: Optional[list[str]] = None,
        limit: int = 15,
    ) -> Optional[Any]:
        """
        Report occurrences of one or more values within specified columns.

        Args:
            values_to_find (Any | list[Any]): Value or list of values to search for.
            columns (list[str] | None): Columns to restrict the search to.
            limit (int): Maximum rows to display in the console.

        Returns:
            pd.DataFrame | None: Summary table of counts and percentages.
        """
        if not isinstance(values_to_find, (list, tuple, set)):
            values_to_find = [values_to_find]

        print(f"--- Occurrences Report for value(s): {values_to_find} ---")
        result_df = self._profiler.find_value_occurrences(values_to_find, columns)
        result_df = result_df.sort_values(by=["Column", "Count"], ascending=[True, False])
        print(result_df.head(limit).to_string(index=False))

        total_rows = len(self._profiler._df)
        print(f"\nTotal rows analysed: {total_rows}")
        if len(result_df) > limit:
            print(f"(showing top {limit} rows)\n")
        print("------------------------------------------\n")
        return result_df

    def show_outliers(
        self,
        columns: Union[str, list[str]],
        method: str = "iqr",
        **kwargs: Any,
    ) -> None:
        """
        Identify and display statistical outliers.

        Args:
            columns (str | list[str]): Column or list of columns to analyse.
            method (str): Detection method ('iqr', 'zscore', etc.).
            **kwargs: Additional method-specific parameters.
        """
        outliers = self._profiler.identify_outliers(columns, method, **kwargs)
        print("--- Outlier Report ---")
        display(outliers)
        print(outliers.shape)

    # ==========================================================================
    # STATISTICS METHODS
    # ==========================================================================

    def show_descriptive_stats(self) -> None:
        """Print classic descriptive statistics (mean, std, quartiles)."""
        df_stats = self._stats.get_descriptive_stats()
        print(f"--- Descriptive Statistics {self.name} ---")
        print(df_stats.to_string())
        print("-----------------------------------------\n")

    def show_skewness(self, columns: Optional[list[str]] = None) -> None:
        """Print skewness for each numerical feature."""
        df_skew = self._stats.get_skewness(columns)
        print(f"--- Skewness of Numerical Columns {self.name} ---")
        print(df_skew.to_string(index=False))
        print("------------------------------------\n")

    def show_normality(self, columns: Optional[list[str]] = None) -> None:
        """Perform D’Agostino–Pearson normality test and print results."""
        df_norm = self._stats.get_normality(columns)
        print(f"--- Normality Test (D’Agostino–Pearson) {self.name} ---")
        print(df_norm.to_string(index=False))
        print("-------------------------------------------\n")

    def plot_correlation_matrix(
        self,
        columns: Optional[list[str]] = None,
        method: str = "pearson",
        dataset_label: Optional[str] = None
    ) -> None:
        """
        Plot a correlation heatmap using the Visualiser.
    
        Delegates computation to the StatisticsCalculator
        and plotting to the Visualiser.
        """
        # Default to analyzer name if label not given
        if dataset_label is None:
            dataset_label = self.name
    
        print(f"--- Plotting {method.capitalize()} Correlation Matrix ({dataset_label}) ---")
        self._visualizer.plot_correlation_matrix(
            stats_calculator=self._stats,
            columns=columns,
            method=method,
            dataset_label=dataset_label,
        )
    
    def plot_covariance_matrix(self, columns: Optional[list[str]] = None) -> None:
        print(f"--- Plotting covariance matrix ({self.name}) ---")
        cov = self._stats.get_covariance_matrix(columns)
        if cov.empty:
            print("No numerical columns available for covariance plot.\n")
            return
        self._visualizer.plot_covariance_heatmap(cov, dataset_label=self.name)

    # ----------------------------------------------------------------------
    # Covariance matrix methods
    # ----------------------------------------------------------------------
    
    def show_covariance_summary(self, columns: Optional[list[str]] = None) -> None:
        """
        Print covariance summary statistics (mean, median, std) for numerical features.
        """
        print(f"--- Covariance Summary {self.name}---")
        cov = self._stats.get_covariance_matrix(columns)
        if cov.empty:
            print("No numerical columns available for covariance analysis.\n")
            return
    
        summary = self._stats.summarize_covariance(cov)
        print(summary.to_string(index=False))
        print("-------------------------------------------\n")

    # ==========================================================================
    # SCHEMA METHODS
    # ==========================================================================

    def set_column_map(self, mapping: Dict[str, str]) -> None:
        """
        Define human-readable aliases for column names.

        Args:
            mapping (dict[str, str]): Mapping of original → display names.
        """
        self._schema.set_mapping(mapping)
        print("Column map updated.\n")

    # ==========================================================================
    # VISUALISER METHODS
    # ==========================================================================

    def plot_bar(
        self,
        x_col: str,
        y_col: Optional[str] = None,
        hue: Optional[str] = None,
        stacked: bool = False,
        colormap: str = "viridis",
        **kwargs
    ) -> None:
        """
        High-level interface for bar or stacked bar charts.
        Passes parameters to the Visualizer for plotting.
        """
        df = self._profiler.get_dataframe()
        self._visualizer.plot_bar(
            df=df,
            x_col=x_col,
            y_col=y_col,
            hue=hue,
            stacked=stacked,
            colormap=colormap,
            **kwargs
        )

    def plot_distribution(self, column_names: Union[str, list[str]]) -> None:
        """Plot distributions (histogram or bar) for one or more columns."""
        print(f"--- Plotting Distribution(s) {self.name}---")
        self._visualizer.plot_distribution(column_names, dataset_label=self.name)

    def plot_scatter(self, x_col: str, y_col: str, hue: Optional[str] = None) -> None:
        """
        Plot a scatter chart showing the relationship between two features.
        Optionally colour points by a third variable (e.g., cluster label).
        """
        print(f"--- Plotting Scatter Plot ({self.name}) ---")
        self._visualizer.plot_scatter(x_col, y_col, dataset_label=self.name, hue=hue)

    def plot_boxplots(
        self,
        numerical_cols: Union[str, list[str]],
        group_by_col: Optional[str] = None,
    ) -> None:
        """
        Create boxplots for one or more numerical columns.

        Args:
            numerical_cols (str | list[str]): Columns to plot.
            group_by_col (str | None): Optional categorical column for grouping.
        """
        print(f"--- Plotting Box Plot(s) {self.name}---")
        self._visualizer.plot_boxplots(numerical_cols, group_by_col, dataset_label=self.name)
    
    def plot_grouped_boxgrid(self, numerical_cols: list[str], group_col: str) -> None:
        """Plot a grid of boxplots for numerical columns grouped by one categorical variable."""
        print(f"--- Plotting grouped boxgrid ({self.name}) ---")
        self._visualizer.plot_grouped_boxgrid(
            numerical_cols, group_col, dataset_label=self.name
        )

    def plot_pairplot(
        self,
        columns: Optional[list[str]] = None,
        hue: Optional[str] = None,
    ) -> None:
        """
        Generate a pairplot (scatter-plot matrix) of selected features.

        Args:
            columns (list[str] | None): Numerical columns to include. Defaults to all.
            hue (str | None): Optional categorical column for colour grouping.

        Example:
            analyzer.plot_pairplot(columns=["Fresh", "Milk", "Grocery"], hue="Region")
        """
        print(f"--- Plotting Pair Plot(s) {self.name}---")
        self._visualizer.plot_pairplot(columns=columns, hue=hue, dataset_label=self.name)

    def plot_pca_scree(self, columns: list[str] | None = None) -> None:
        """
        Compute and plot the PCA scree plot for selected numerical columns.
        """
        print(f"--- Plotting PCA scree plot ({self.name}) ---")
        pca_df = self._stats.get_pca_components(columns=columns)
        if pca_df.empty:
            return
        self._visualizer.plot_pca_scree(pca_df, dataset_label=self.name)

    # ==============================================================================
    # Extended visualization orchestration methods
    # ==============================================================================
    
    def plot_pca_scatter(self, columns: list[str] | None = None) -> None:
        """Compute and plot PCA scatter (PC1 vs PC2)."""
        print(f"--- Plotting PCA scatter ({self.name}) ---")
        pca_df = self._stats.get_pca_scatter_data(columns=columns)
        if pca_df.empty:
            return
        self._visualizer.plot_pca_scatter(pca_df, dataset_label=self.name)
    
    
    def plot_k_distance(self, columns: list[str] | None = None, k: int = 4) -> None:
        """Compute and plot k-distance curve."""
        print(f"--- Plotting k-distance plot ({self.name}) ---")
        distances_df = self._stats.get_k_distance(columns=columns, k=k)
        if distances_df.empty:
            return
        self._visualizer.plot_k_distance(distances_df, dataset_label=self.name)
    
    
    def plot_dendrogram(self, columns: list[str] | None = None, method: str = "ward") -> None:
        """Compute and plot hierarchical clustering dendrogram."""
        print(f"--- Plotting dendrogram ({self.name}) ---")
        linkage_matrix = self._stats.get_hierarchical_linkage(columns=columns, method=method)
        if linkage_matrix.size == 0:
            return
        self._visualizer.plot_dendrogram(linkage_matrix, dataset_label=self.name)
    
    
    def plot_tsne(self, columns: list[str] | None = None) -> None:
        """Compute and plot t-SNE embedding."""
        print(f"--- Plotting t-SNE embedding ({self.name}) ---")
        tsne_df = self._stats.get_tsne_embedding(columns=columns)
        if tsne_df.empty:
            return
        self._visualizer.plot_tsne_embedding(tsne_df, dataset_label=self.name)
    
    
    def plot_umap(self, columns: list[str] | None = None) -> None:
        """Compute and plot UMAP embedding."""
        print(f"--- Plotting UMAP embedding ({self.name}) ---")
        umap_df = self._stats.get_umap_embedding(columns=columns)
        if umap_df.empty:
            return
        self._visualizer.plot_umap_embedding(umap_df, dataset_label=self.name)

    def plot_elbow(self, columns: Optional[list[str]] = None, k_range: range = range(1, 11)) -> None:
        """
        Compute and plot the K-Means elbow curve, and print the optimal k suggestion.
        """
        print(f"--- Computing K-Means elbow curve ({self.name}) ---")
    
        X = self._profiler._df[columns].select_dtypes(include=["number"]).dropna().values
        elbow_df = self._stats.compute_elbow_curve(X, k_range)
    
        # NEW: auto-detect elbow
        optimal_k = self._stats.elbow_by_second_diff(elbow_df["k"], elbow_df["inertia"])
        print(f"Suggested optimal number of clusters (via curvature): k = {optimal_k}")
    
        # Plot it
        self._visualizer.plot_elbow_curve(elbow_df, dataset_label=self.name)

    def plot_silhouette(self, columns: list[str], k_range=range(2, 11)):
        """
        High-level method to compute and plot silhouette scores for different k.
        """
        print(f"--- Computing silhouette scores ({self.name}) ---")
        X = self._profiler._df[columns].select_dtypes(include=["number"]).dropna().values
        df_scores = self._stats.compute_silhouette_scores(X, k_range)
        k_best, max_score = df_scores.loc[df_scores["silhouette_score"].idxmax(), ["k", "silhouette"]]
        print(f"Maximumn silhouette score for k={k_best}: {max_score:.3f}")
        self._visualizer.plot_silhouette_scores(df_scores, dataset_label=self.name)
    
    # ==========================================================================
    # CLEANER METHODS
    # ==========================================================================

    def clean_replace_values(
        self,
        columns: list[str],
        value_to_replace: Any,
        new_value: Any = np.nan,
    ) -> None:
        """Replace specified values in given columns."""
        print("--- Replacing Values ---")
        self._cleaner.replace_values(columns, value_to_replace, new_value)

    def clean_coerce_numeric(self, columns: list[str]) -> None:
        """Force columns to numeric dtype, coercing errors to NaN."""
        print("--- Coercing to Numeric ---")
        self._cleaner.coerce_to_numeric(columns)

    # ==========================================================================
    # REPORT GENERATION
    # ==========================================================================

    def export_plots_to_word(self, filename: str | None = None) -> None:
        """
        Coordinate report generation by delegating to the ReportGenerator.
    
        Args:
            filename (str | None): Optional target filename. If not provided,
                                   the ReportGenerator will derive one based on
                                   the analyzer's name and timestamp.
        """
        print("--- Exporting Plots to Word Document ---")
    
        plot_cache = self._visualizer.get_plot_cache()
        if not plot_cache:
            print("No plots generated yet.")
            return
    
        self._report_generator.create_word_document(
            plot_cache=plot_cache,
            filename=filename,
            analyzer_name=self.name,
            df_preview=getattr(self._profiler, "_df", None)
        )

    # ==========================================================================
    # VERSION / SESSION INFO
    # ==========================================================================

    def _save_session_metadata(self) -> None:
        """Write session metadata (timestamp, version, dataset info) to a JSON log."""
        meta = {
            "suite": "EDA Suite",
            "suite_version": self.suite_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "version_tag": self.version_tag,
            "data_shape": getattr(self._profiler._df, "shape", None),
            "columns": list(getattr(self._profiler._df, "columns", [])),
        }
        path = os.path.join(self._log_dir, f"{self.version_tag}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        print(f"[LOG] EDA session metadata saved → {path}")

    def show_session_info(self) -> None:
        """Display suite version, session ID, and log-file location."""
        print("--- EDA Suite Session Info ---")
        print(f"Suite Version: {self.suite_version}")
        print(f"Session ID:    {self.session_id}")
        print(f"Created At:    {self.created_at}")
        print(f"Version Tag:   {self.version_tag}")
        print(f"Logs saved to: {self._log_dir}")
        print("-----------------------------------\n")

    # ======================================================================
    # FACTORY CONSTRUCTOR
    # ======================================================================
    @classmethod
    def from_dataframe(cls, df, name: str = "EDA session") -> "EDAAnalyzer":
        """
        Factory constructor that automatically builds a fully configured EDAAnalyzer
        (Profiler, Stats, Visualizer, SchemaManager, Cleaner, ReportGenerator)
        from a given pandas DataFrame.

        This replaces the need to manually create and wire an EdaContainer.

        Args:
            df (pd.DataFrame): The dataset to analyse.
            name (str): Optional human-readable session name (e.g. 'Raw data').

        Returns:
            EDAAnalyzer: A ready-to-use, fully configured analyzer instance.
        """
        from .profiler import DataProfiler
        from .statistics import StatisticsCalculator
        from .visualizer import Visualizer
        from .schema import SchemaManager
        from .cleaner import DataCleaner
        from .report_generator import ReportGenerator

        # --- Step 1: Initialise shared components ---
        schema = SchemaManager(df)
        profiler = DataProfiler(df)
        stats = StatisticsCalculator(df)
        visualizer = Visualizer(df, schema)
        cleaner = DataCleaner(df)
        report_generator = ReportGenerator()

        # --- Step 2: Construct the analyzer with all dependencies injected ---
        return cls(
            profiler=profiler,
            stats=stats,
            visualizer=visualizer,
            schema=schema,
            cleaner=cleaner,
            report_generator=report_generator,
            name=name,
        )




