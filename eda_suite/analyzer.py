# ==============================================================================
# analyzer.py
#
# This module provides the EDAAnalyzer class, which serves as the primary
# user interface for the entire EDA Suite. It employs the Facade design pattern
# to provide a simple, clean, and unified set of commands to interact with the
# more complex, underlying subsystem of specialist components (Profiler, Cleaner, etc.).
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

from typing import Dict, Union, List, Any, Optional
import numpy as np
import uuid
import json
import os
from datetime import datetime

# Import the specialist components that this facade will manage.
# The type hints (e.g., : DataProfiler) are crucial for code clarity and static analysis.
from .profiler import DataProfiler
from .statistics import StatisticsCalculator
from .visualizer import Visualizer
from .schema import SchemaManager
from .cleaner import DataCleaner
from .report_generator import ReportGenerator
import io
import os

class EDAAnalyzer:
    """
    Facade class that orchestrates the entire Exploratory Data Analysis process.
    
    This class acts as a single, convenient entry point for all EDA tasks. It does
    not contain complex logic itself; instead, it delegates tasks to the appropriate
    specialist component that was injected upon initialisation.

    The EDAAnalyzer integrates profiling, visualisation, and statistical
    heuristics to provide quick insights into dataset structure.
    """

    def __init__(
        self,
        profiler: DataProfiler,
        stats: StatisticsCalculator,
        visualizer: Visualizer,
        schema: SchemaManager,
        cleaner: DataCleaner,
        report_generator: ReportGenerator,
    ):
        """
        Initialises the EDAAnalyzer with all its required dependencies.
        
        This constructor is designed for Dependency Injection. It receives fully-formed
        specialist objects from a container and stores them as internal attributes.
        It essentially assembles the toolbox for the analyst to use.
        """
        self._profiler = profiler
        self._stats = stats
        self._visualizer = visualizer
        self._schema = schema
        self._cleaner = cleaner
        self._report_generator = report_generator
        # This dictionary stores any manual overrides for column types, ensuring
        # the user's domain knowledge can correct the automated heuristics.
        self._type_overrides = {}

            # -----------------------------------------------------------
        # Version tracking metadata
        # -----------------------------------------------------------
        self.session_id = uuid.uuid4().hex[:8]
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Optionally import package version if defined in __init__.py
        try:
            from eda_suite import __version__ as suite_version
        except ImportError:
            suite_version = "unknown"

        self.suite_version = suite_version
        # Replace colons with hyphens for Windows-safe filenames
        safe_timestamp = self.created_at.replace(":", "-").replace(" ", "_")
        self.version_tag = f"EDA_{safe_timestamp}_{self.session_id}"

        # Create a .logs folder if it doesn't exist
        self._log_dir = os.path.join(os.getcwd(), ".logs")
        os.makedirs(self._log_dir, exist_ok=True)

        # Save session metadata immediately
        self._save_session_metadata()

    # ==========================================================================
    # --- Profiler Methods: For Diagnosing and Inspecting Data ---
    # These methods analyse the data's state without modifying it.
    # ==========================================================================

    def set_column_type(self, column: str, new_type: str):
        """
        Manually overrides the heuristic type for a column. This allows the user
        to enforce their domain knowledge over the automated classification.
        
        Args:
            column (str): The name of the column to override.
            new_type (str): The desired type ('numerical', 'categorical', or 'id').
        """
        valid_types = ['numerical', 'categorical', 'id']
        if new_type.lower() not in valid_types:
            print(f"Error: Invalid type '{new_type}'. Must be one of {valid_types}")
            return
        self._type_overrides[column] = new_type.lower()
        print(f"Override set: Column '{column}' will now be treated as '{new_type.lower()}'.\n")

    def show_profile(self):
        """Prints a summary profile of the DataFrame using advanced heuristics."""
        # Delegate the heavy lifting of analysis to the profiler, passing any overrides.
        profile = self._profiler.get_summary(overrides=self._type_overrides)
        
        # The facade's second job: format the raw result for clean presentation.
        print("--- DataFrame Heuristic Profile ---")
        print(f"Shape: {profile['shape'][0]} rows, {profile['shape'][1]} columns")
        print("\nNumerical Columns (likely continuous):")
        print(f"  {profile['numerical_columns (heuristic)']}")
        print("\nCategorical Columns (likely discrete):")
        print(f"  {profile['categorical_columns (heuristic)']}")
        print("\nID Columns (high cardinality):")
        print(f"  {profile['id_columns (heuristic)']}")
        print("-----------------------------------\n")

    def show_mixed_type_report(self):
        """Prints a report of columns that contain multiple data types."""
        report = self._profiler.get_mixed_type_report()
        print("--- Mixed Data Type Report ---")
        if not report:
            print("No columns with mixed data types found.")
        else:
            print("Warning: The following columns contain multiple data types:")
            for col, types in report.items():
                print(f"  - Column '{col}': contains types {types}")
        print("------------------------------\n")

    def show_missing_values(self):
        """Prints a report of standard missing values (NaNs)."""
        missing_df = self._profiler.get_missing_values()
        print("--- Missing Values Report ---")
        if missing_df.empty:
            print("No missing values found.")
        else:
            # Use .to_string() to ensure the full report is printed without truncation.
            print(missing_df.to_string())
        print("---------------------------\n")

    def show_value_occurrences(
        self,
        values_to_find: any,
        columns: list[str] | None = None,
        limit: int = 15
    ):
        """
        Prints a report of where specific value(s) occur across columns,
        with counts and percentages.
        """
        # Normalise input to list
        if not isinstance(values_to_find, (list, tuple, set)):
            values_to_find = [values_to_find]
    
        print(f"--- Occurrences Report for value(s): {values_to_find} ---")
        result_df = self._profiler.find_value_occurrences(values_to_find, columns)
    
        # Clean display
        result_df = result_df.sort_values(by=["Column", "Count"], ascending=[True, False])
        print(result_df.head(limit).to_string(index=False))
    
        total_rows = len(self._profiler._df)
        print(f"\nTotal rows analysed: {total_rows}")
        if len(result_df) > limit:
            print(f"(showing top {limit} rows)\n")
        print("------------------------------------------\n")
    
        return result_df



    def show_outliers(self, 
                            columns: Union[str, List[str]], 
                            method: str = 'iqr', 
                            **kwargs):
        """
        Identifies outliers in specified columns of a DataFrame using a chosen statistical method.
        """
        outlier_rows_df = self._profiler.identify_outliers(columns, method, **kwargs) 
        print(f"--- Occurrences of outliers: ---")
        display(outlier_rows_df)
        print(outlier_rows_df.shape)

    # ==========================================================================
    # --- Statistics Methods: For Quantitative Analysis ---
    # ==========================================================================

    def show_descriptive_stats(self):
        """Prints descriptive statistics (mean, std, etc.) for numerical columns."""
        stats_df = self._stats.get_descriptive_stats()
        print("--- Descriptive Statistics (Numerical) ---")
        print(stats_df.to_string())
        print("-----------------------------------------\n")

    def show_skewness(self, columns: list[str] | None = None):
        """Prints skewness values for numerical columns."""
        df_skew = self._stats.get_skewness(columns)
        print("--- Skewness of Numerical Columns ---")
        print(df_skew.to_string(index=False))
        print("------------------------------------\n")

    def show_normality(self, columns: list[str] | None = None):
        """Prints p-values from the normality test for numerical columns."""
        df_norm = self._stats.get_normality(columns)
        print("--- Normality Test (D’Agostino–Pearson) ---")
        print(df_norm.to_string(index=False))
        print("-------------------------------------------\n")
    
    # ==========================================================================
    # --- Schema Methods: For Managing Column Metadata ---
    # ==========================================================================

    def set_column_map(self, mapping: Dict[str, str]):
        """Sets human-readable alias names for columns to improve plot/report readability."""
        self._schema.set_mapping(mapping)
        print("Column map updated.\n")

    # ==========================================================================
    # --- Visualizer Methods: For Graphical Analysis ---
    # These are direct pass-throughs to the Visualiser component.
    # ==========================================================================

    def plot_distribution(self, column_names: Union[str, List[str]]):
        """Delegates the plotting of data distributions (histograms or bar charts)."""
        print("--- Plotting Distribution(s) ---")
        self._visualizer.plot_distribution(column_names)

    def plot_scatter(self, x_col: str, y_col: str):
        """Delegates the plotting of a scatter plot to show relationships."""
        print("--- Plotting Scatter Plot ---")
        self._visualizer.plot_scatter(x_col, y_col)

    def plot_boxplots(self, numerical_cols: Union[str, List[str]], group_by_col: Optional[str] = None):
        """Delegates the plotting of box plots for distribution analysis."""
        print("--- Plotting Box Plot(s) ---")
        self._visualizer.plot_boxplots(numerical_cols, group_by_col)

    def plot_pairplot(self, columns=None, hue=None):
    """Plots a pair plot to visualise relationships between numerical features."""
    print("--- Generating Pair Plot ---")
    self._visualizer.plot_pairplot(columns, hue)

    # ==========================================================================
    # --- Cleaner Methods: For Modifying the DataFrame ---
    # These methods alter the state of the underlying data.
    # ==========================================================================

    def clean_replace_values(self, columns: List[str], value_to_replace: Any, new_value: Any = np.nan):
        """Delegates the replacement of specific values within columns."""
        print("--- Replacing Values ---")
        self._cleaner.replace_values(columns, value_to_replace, new_value)

    def clean_coerce_numeric(self, columns: List[str]):
        """Delegates the conversion of columns to a numeric data type."""
        print("--- Coercing to Numeric ---")
        self._cleaner.coerce_to_numeric(columns)

    # ==========================================================================
    # --- Report Generator Methods: For Creating Reports ---
    # ==========================================================================

    def export_plots_to_word(self, filename: str = "eda_report.docx"):
        """
        Gathers all plots generated during the session and exports them to a
        Word document.
        """
        print("--- Exporting Plots to Word Document ---")
        
        # Step 1: The Analyzer gets the cache from its internal Visualizer tool.
        # The user doesn't have to worry about this part.
        plot_cache = self._visualizer.get_plot_cache()

        # A quick check to make sure there's something to report.
        if not plot_cache:
            print("No plots have been generated yet. Nothing to export.")
            return
        
        # Step 2: The Analyzer delegates the report creation to its internal
        # ReportGenerator tool, passing the cache and filename correctly.
        self._report_generator.create_word_document(plot_cache, filename)

    # ==========================================================================
    # --- Version tracking ---
    # ==========================================================================

    def _save_session_metadata(self):
        """Logs key session metadata (timestamp, version, dataset info) to a JSON file."""
        metadata = {
            "suite": "EDA Suite",
            "suite_version": self.suite_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "version_tag": self.version_tag,
            "data_shape": getattr(self._profiler._df, "shape", None),
            "columns": list(getattr(self._profiler._df, "columns", [])),
        }

        filename = os.path.join(self._log_dir, f"{self.version_tag}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"[LOG] EDA session metadata saved → {filename}")

    def show_session_info(self):
        """Displays session version and metadata."""
        print(f"--- EDA Suite Session Info ---")
        print(f"Suite Version: {self.suite_version}")
        print(f"Session ID:    {self.session_id}")
        print(f"Created At:    {self.created_at}")
        print(f"Version Tag:   {self.version_tag}")
        print(f"Logs saved to: {self._log_dir}")
        print("-----------------------------------\n")


