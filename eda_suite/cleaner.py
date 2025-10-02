# ==============================================================================
# cleaner.py
#
# This module provides the DataCleaner class. Its single responsibility is to
# perform data cleaning and transformation operations. Unlike the other
# components of the EDA Suite which are read-only, the methods within this
# class are designed to modify the underlying DataFrame in-place.
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

import pandas as pd
import numpy as np
from typing import List, Any

class DataCleaner:
    """
    Handles data cleaning and transformation tasks on a DataFrame.
    
    The methods herein provide a clean interface for common data preparation
    steps, such as standardising missing values and correcting data types.
    Operations are performed in-place on the DataFrame provided during initialisation.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialises the DataCleaner with a direct reference to a DataFrame.
        
        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to be cleaned. Note that
                                      changes made by this class will affect this
                                      original DataFrame object.
        """
        self._df = dataframe

    def replace_values(self, columns: List[str], value_to_replace: Any, new_value: Any = np.nan):
        """
        Replaces a specific value across multiple columns.
        
        This is the primary method for standardising custom null values
        (e.g., 'N.U.', 'Not Applicable', -999) into the standard pandas
        missing value representation, `np.nan`.
        
        Args:
            columns (List[str]): A list of column names to perform the replacement on.
            value_to_replace (Any): The specific value to be found and replaced.
            new_value (Any, optional): The value to replace with. Defaults to np.nan.
        """
        print(f"Replacing '{value_to_replace}' with '{new_value}' in columns: {columns}...")
        # We iterate through the provided column list to apply the change individually.
        for col in columns:
            # This is a crucial safety check to ensure the code doesn't crash if
            # the user supplies a column name that doesn't exist in the DataFrame.
            if col in self._df.columns:
                # --- Robust Assignment Pattern ---
                # This line is the recommended way to modify a column in pandas.
                # It performs the `.replace()` operation, which returns a new pandas Series
                # with the changes, and then explicitly assigns that new Series back to
                # its place in the DataFrame. This avoids the ambiguous 'SettingWithCopyWarning'.
                self._df[col] = self._df[col].replace(value_to_replace, new_value)
            else:
                print(f"Warning: Column '{col}' not found. Skipping.")
        print("Replacement complete.\n")

    def coerce_to_numeric(self, columns: List[str]):
        """
        Forcefully converts the data type of specified columns to a numeric type.
        
        This method is essential for correcting columns that should be numerical but are
        currently stored as 'object' dtype due to the presence of non-numeric characters.
        
        Args:
            columns (List[str]): A list of column names to be converted.
        """
        print(f"Attempting to convert columns to numeric: {columns}...")
        for col in columns:
            if col in self._df.columns:
                # The `pd.to_numeric` function is the standard for type conversion.
                # The `errors='coerce'` argument is incredibly powerful for data cleaning.
                # It tells pandas: "Try to convert every value to a number. If you
                # encounter a value that you can't convert (e.g., the string 'N.U.'),
                # don't raise an error. Instead, silently replace it with NaN."
                # This line also uses the robust assignment pattern.
                self._df[col] = pd.to_numeric(self._df[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found. Skipping.")
        print("Conversion complete.\n")