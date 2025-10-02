# ==============================================================================
# profiler.py
#
# This module contains the DataProfiler class, which serves as the primary
# investigative tool in the EDA Suite. Its responsibility is to analyse the
# structure and contents of a DataFrame and report its findings without
# altering the original data in any way. It forms the diagnostic core of the
# analysis workflow.
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

import pandas as pd
from typing import List, Dict, Any

class DataProfiler:
    """
    Analyses the structure, data types, and missing values of a DataFrame.
    
    This class provides methods to gain a high-level understanding of the dataset,
    including intelligent, heuristic-based classification of columns and the
    detection of common data quality issues.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialises the DataProfiler with the target DataFrame.
        
        Args:
            dataframe (pd.DataFrame): The dataset to be analysed.
        """
        # The underscore prefix denotes an internal attribute, intended for use
        # only within the methods of this class.
        self._df = dataframe

    def get_summary(self, overrides: dict = None) -> Dict[str, Any]:
        """
        Provides an advanced summary by applying heuristics to classify columns.
        This method intelligently attempts to distinguish between continuous numerical,
        discrete categorical, and high-cardinality ID columns.

        Args:
            overrides (dict, optional): A dictionary to manually force a classification
                                        for specific columns, bypassing the heuristics. 
                                        Defaults to None.
        
        Returns:
            Dict[str, Any]: A dictionary containing a detailed profile of the DataFrame.
        """
        total_rows = len(self._df)
        if overrides is None:
            overrides = {}

        # --- Graceful Fallback for Empty DataFrames ---
        # This check prevents errors if the DataFrame has no rows.
        if total_rows == 0:
            return {
                "shape": self._df.shape, "column_names": self._df.columns.tolist(),
                "numerical_columns (heuristic)": [], "categorical_columns (heuristic)": [],
                "id_columns (heuristic)": [],
            }
        
        # Initialise lists to hold the classified column names.
        numerical_cols, categorical_cols, id_cols = [], [], []

        # Iterate over each column to apply our classification logic individually.
        for col in self._df.columns:
            # --- Manual Override Check ---
            # First, check if the user has provided a manual override for this column.
            # This ensures that human domain knowledge always takes precedence.
            if col in overrides:
                col_type = overrides[col]
                if col_type == 'numerical': numerical_cols.append(col)
                elif col_type == 'categorical': categorical_cols.append(col)
                elif col_type == 'id': id_cols.append(col)
                continue  # Skip the heuristic logic for this column.

            # --- Heuristic Logic Begins ---
            dtype = self._df[col].dtype
            nunique = self._df[col].nunique()
            
            # Branch 1: The column has a numeric data type (int, float).
            if pd.api.types.is_numeric_dtype(dtype):
                # Heuristic 1.1: Check for ID-like columns. If the number of unique values
                # is equal to the number of rows (or very close to it), it's likely a
                # unique identifier (e.g., a primary key) and not a continuous variable.
                if nunique == total_rows or (nunique / total_rows) > 0.95:
                    id_cols.append(col)
                # Heuristic 1.2: Check for low-cardinality integers. If an integer column
                # has very few unique values (e.g., a 'Star Rating' of 1-5), it behaves
                # as a categorical variable and should be treated as such.
                elif nunique < 25:
                    categorical_cols.append(col)
                # Heuristic 1.3: If it's numeric but not an ID or low-cardinality,
                # we can safely classify it as a continuous numerical column.
                else:
                    numerical_cols.append(col)
            # Branch 2: The column has a non-numeric type (e.g., object, datetime).
            # These are almost always treated as categorical in nature for EDA purposes.
            else:
                categorical_cols.append(col)

        # Collate all the analysed information into a single report dictionary.
        return {
            "shape": self._df.shape, "column_names": self._df.columns.tolist(),
            "numerical_columns (heuristic)": numerical_cols,
            "categorical_columns (heuristic)": categorical_cols,
            "id_columns (heuristic)": id_cols,
        }

    def get_missing_values(self) -> pd.DataFrame:
        """Calculates the count and percentage of missing values (NaNs) for each column."""
        # A standard, efficient pandas idiom to count nulls in each column.
        missing_counts = self._df.isnull().sum()
        # Calculate the percentage to provide better context, especially for large datasets.
        missing_percentages = (missing_counts / len(self._df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        })
        
        # We only return columns that actually contain missing values to keep the
        # report concise and actionable. The results are sorted for clarity.
        return missing_df[missing_df['missing_count'] > 0].sort_values(
            by='missing_count', ascending=False
        )

    def find_value_occurrences(self, value_to_find: any) -> pd.DataFrame:
        """
        Finds all occurrences of a user-specified value (e.g., 'N.U.', -999, 'Not Available').
        This is useful for identifying custom null values that `isnull()` would not detect.
        """
        # This performs a boolean comparison across the entire DataFrame. The result is a
        # DataFrame of the same shape, containing True/False values.
        occurrences = (self._df == value_to_find)
        
        # A clever trick in pandas: summing a boolean series/DataFrame counts the `True` values.
        occurrence_counts = occurrences.sum()
        
        occurrence_percentages = (occurrence_counts / len(self._df)) * 100
        
        occurrence_df = pd.DataFrame({
            'occurrence_count': occurrence_counts,
            'occurrence_percentage': occurrence_percentages
        })
        
        # As before, we filter for non-zero counts to produce a clean report.
        return occurrence_df[occurrence_df['occurrence_count'] > 0].sort_values(
            by='occurrence_count', ascending=False
        )
    
    def get_mixed_type_report(self) -> dict:
        """
        Inspects each column for multiple underlying Python data types,
        which often indicates data entry errors or data corruption.
        """
        mixed_type_info = {}
        for col in self._df.columns:
            # This chain performs a detailed inspection:
            # 1. `.dropna()`: Ignore null values, as they don't have a type.
            # 2. `.apply(type)`: Apply the built-in `type()` function to each value.
            # 3. `.unique()`: Get an array of the unique types found (e.g., [str, int]).
            types_in_col = self._df[col].dropna().apply(type).unique()
            
            # If more than one type was found, we flag the column as problematic.
            if len(types_in_col) > 1:
                # We use `t.__name__` to get a clean string like 'str' instead of "<class 'str'>".
                mixed_type_info[col] = [t.__name__ for t in types_in_col]
        return mixed_type_info