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

# Import the specialist components that this facade will manage.
# The type hints (e.g., : DataProfiler) are crucial for code clarity and static analysis.
from .profiler import DataProfiler
from .statistics import StatisticsCalculator
from .visualizer import Visualizer
from .schema import SchemaManager
from .cleaner import DataCleaner
from .report_generator import ReportGenerator
import io

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

    def show_value_occurrences(self, value_to_find: any):
        """Prints a report of a specific custom value's occurrences (e.g., 'N.U.', -999)."""
        occurrence_df = self._profiler.find_value_occurrences(value_to_find)
        print(f"--- Occurrences Report for value: '{value_to_find}' ---")
        if occurrence_df.empty:
            print(f"No occurrences of '{value_to_find}' found.")
        else:
            print(occurrence_df.to_string())
        print("------------------------------------------\n")

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
