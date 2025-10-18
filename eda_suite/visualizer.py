# ==============================================================================
# visualiser.py
#
# This module provides the Visualiser class, a core component of the EDA Suite.
# Its single responsibility is to generate high-quality data visualisations
# based on the provided DataFrame. It is designed to be decoupled from other
# components, making decisions based on the data's current state (e.g., its dtype).
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .schema import SchemaManager
from typing import Union, List, Optional
import io

class Visualizer:
    """
    Handles the creation of plots and charts for Exploratory Data Analysis.
    
    This class encapsulates all plotting logic, using libraries like Matplotlib
    and Seaborn. It also features a plot caching mechanism to support the
    generation of reports.
    """

    def __init__(self, dataframe: pd.DataFrame, schema_manager: SchemaManager):
        """
        Initialises the Visualiser with the dataset and a schema manager.
        
        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to be analysed.
            schema_manager (SchemaManager): An object that manages the mapping
                                           between original and display column names.
        """
        # The leading underscore indicates these are internal attributes, not to be
        # modified directly from outside the class.
        self._df = dataframe
        self._schema = schema_manager
        
        # The plot cache is a list that will store every generated plot in memory.
        # This is a crucial feature for the report generation functionality.
        self._plot_cache: List[io.BytesIO] = []
        
        # Set a professional and aesthetically pleasing default theme for all plots.
        sns.set_theme(style="whitegrid")

    def get_plot_cache(self) -> List[io.BytesIO]:
        """
        A simple getter method to retrieve the cached plots.
        
        Returns:
            List[io.BytesIO]: A list of plot images stored as in-memory binary objects.
        """
        return self._plot_cache

    def _save_plot_to_cache(self):
        """
        Saves the current matplotlib figure to an in-memory binary buffer.
        
        This is a powerful technique that avoids writing temporary files to the disk.
        The plot is saved as a PNG into a BytesIO object, which behaves like a file
        but exists only in memory.
        """
        # Create a binary buffer to hold the image data.
        img_buffer = io.BytesIO()
        
        # Save the current figure into the buffer.
        # The bbox_inches='tight' argument is a pro-tip to trim any excess
        # whitespace around the plot, making it look cleaner in reports.
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        
        # After writing, the buffer's 'cursor' is at the end. We must rewind it
        # to the beginning so that other functions (like python-docx) can read it.
        img_buffer.seek(0)
        
        # Add the completed image buffer to our cache list.
        self._plot_cache.append(img_buffer)

    def plot_distribution(self, column_names: Union[str, List[str]]):
        """
        Plots the distribution of a single column or multiple columns.
        
        This method is polymorphic; it intelligently handles either a single string
        or a list of strings as input. It then determines the best plot type
        based on the column's data type (dtype).
        """
        # Standardise the input: if a single string is provided, wrap it in a list.
        # This allows the rest of the function to operate on a list consistently.
        if isinstance(column_names, str):
            columns_to_plot = [column_names]
        else:
            columns_to_plot = column_names

        # Loop through each requested column and generate a separate plot.
        for col_name in columns_to_plot:
            # Use the schema manager to get the correct underlying column name
            # and the human-readable display name for plot labels.
            original_col_name = self._schema.get_original_name(col_name)
            display_col_name = self._schema.get_display_name(original_col_name)

            # Create a new figure for each plot to ensure they are distinct.
            plt.figure(figsize=(10, 6))
            
            # --- Core Logic: Select plot type based on data type ---
            # For continuous numerical data, a histogram is the ideal visualisation.
            if self._df[original_col_name].dtype in ['int64', 'float64']:
                sns.histplot(self._df[original_col_name], kde=True)
                plt.title(f'Distribution of {display_col_name}')
            # For discrete categorical data, a count plot (bar chart) is appropriate.
            else:
                sns.countplot(y=self._df[original_col_name], order=self._df[original_col_name].value_counts().index)
                plt.title(f'Frequency of {display_col_name}')
            
            # Set labels using the human-readable display name.
            plt.xlabel(display_col_name)
            plt.ylabel('Count' if self._df[original_col_name].dtype in ['int64', 'float64'] else '')
            
            # Before displaying the plot, save a copy to our cache for later use.
            self._save_plot_to_cache()  # Step 1: Save the plot to our report cache
            plt.show()                  # Step 2: Display the plot in the notebook
            plt.close()                 # Step 3: Close the figure to free up memory

    def plot_scatter(self, x_col: str, y_col: str):
        """
        Creates a scatter plot to visualize the relationship between two numerical columns.
        This function will gracefully skip plotting if there is no valid data.
        """
        print(f"--- Plotting Scatter Plot: {y_col} vs. {x_col} ---")
        original_x = self._schema.get_original_name(x_col)
        original_y = self._schema.get_original_name(y_col)
        
        display_x = self._schema.get_display_name(original_x)
        display_y = self._schema.get_display_name(original_y)

        # Create a temporary DataFrame with just the two columns and drop rows
        # where either x or y is a missing value (NaN).
        plot_data = self._df[[original_x, original_y]].dropna()

        # Check if any data remains to be plotted after removing missing values.
        if plot_data.empty:
            print(f"SKIPPING PLOT: No overlapping data points found for '{display_y}' vs. '{display_x}'.")
            return # Exit the function to prevent the crash

        plt.figure(figsize=(10, 6))
        
        # The plotting call now uses the cleaned 'plot_data' DataFrame
        sns.scatterplot(
            x=plot_data[original_x],
            y=plot_data[original_y],
            color='red',
            s=20,
            alpha=0.5,
        )

        plt.title(f'{display_y} vs. {display_x}')
        plt.xlabel(display_x)
        plt.ylabel(display_y)
        
        # Cache and show the plot.
        self._save_plot_to_cache()  # Step 1: Save the plot to our report cache
        plt.show()                  # Step 2: Display the plot in the notebook
        plt.close()                 # Step 3: Close the figure to free up memory

    def plot_boxplots(self, numerical_cols: Union[str, List[str]], group_by_col: Optional[str] = None):
        """
        Generates box plots to show distributions and detect outliers.
        - If 'group_by_col' is provided, it compares the distribution of a numerical
          column across different categories.
        - Otherwise, it compares the distributions of multiple numerical columns.
        - Rotates x-axis labels only if multiple columns are plotted without grouping.
        """
        if isinstance(numerical_cols, str):
            cols_to_plot = [numerical_cols]
        else:
            cols_to_plot = numerical_cols

        original_cols = [self._schema.get_original_name(col) for col in cols_to_plot]

        # Scenario 1: Group a numerical variable by a categorical variable.
        if group_by_col:
            original_group_col = self._schema.get_original_name(group_by_col)
            display_group_col = self._schema.get_display_name(original_group_col)
            
            for original_col, display_col in zip(original_cols, cols_to_plot):
                plt.figure(figsize=(12, 7))
                sns.boxplot(data=self._df, x=original_group_col, y=original_col)
                plt.title(f'Distribution of {display_col} by {display_group_col}')
                plt.xlabel(display_group_col)
                plt.ylabel(display_col)
                # Rotation is generally useful here as category names can be long.
                plt.xticks(rotation=45)
                self._save_plot_to_cache()  # Step 1: Save the plot to our report cache
                plt.show()                  # Step 2: Display the plot in the notebook
                plt.close()                 # Step 3: Close the figure to free up memory
        # Scenario 2: Compare multiple numerical variables side-by-side.
        else:
            plt.figure(figsize=(12, 7))
            
            # To plot multiple columns with Seaborn, it's best to 'melt' the DataFrame.
            # This converts it from a 'wide' format to a 'long' format.
            # E.g., from columns 'Temp', 'Press' to one 'Channel' column and one 'Value' column.
            df_melted = self._df[original_cols].melt(var_name='Channel', value_name='Value')

            # Before plotting, check if there are any valid (non-NaN) data points.
            if df_melted['Value'].notna().sum() == 0:
                # If there's nothing to plot, print a warning and exit the function.
                print(f"SKIPPING PLOT: No valid data found for column(s): {', '.join(cols_to_plot)}")
                plt.close()  # Close the empty figure to free up memory
                return

            # Use the display names for the channels on the plot's x-axis.
            display_name_map = {orig: self._schema.get_display_name(orig) for orig in original_cols}
            df_melted['Channel'] = df_melted['Channel'].map(display_name_map)

            sns.boxplot(data=df_melted, x='Channel', y='Value')
            plt.title(f'Distributions of Selected Columns')
            plt.xlabel('Columns')
            plt.ylabel('Value')

            # Conditional formatting: Only rotate labels if needed to prevent overlap.
            if len(cols_to_plot) > 1:
                plt.xticks(rotation=45)
            
            self._save_plot_to_cache()  # Step 1: Save the plot to our report cache
            plt.show()                  # Step 2: Display the plot in the notebook
            plt.close()                 # Step 3: Close the figure to free up memory

    def plot_pairplot(self, columns=None, hue=None):
        """Creates a pair plot (scatterplot matrix) for given columns."""
        if columns is None:
            columns = self._df.select_dtypes(include=['number']).columns
        sns.pairplot(self._df[columns], hue=hue, diag_kind="kde", corner=True)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

    def plot_correlation_matrix(
        self,
        stats_calculator,
        columns: list[str] | None = None,
        method: str = "pearson",
    ):
        """
        Plots a heatmap of the correlation matrix.
        The computation is delegated to the StatisticsCalculator for consistency.
        """
        corr = stats_calculator.get_correlation_matrix(columns=columns, method=method)
    
        if corr.empty:
            print("No numerical columns found for correlation plot.")
            return
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=14, pad=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
    
        self._save_plot_to_cache()
        plt.show()
        plt.close()

    def plot_covariance_heatmap(self, cov_matrix: pd.DataFrame, title: str = "Covariance Matrix") -> None:
        """
        Plot a heatmap of the covariance matrix.

        Args:
            cov_matrix: Covariance matrix DataFrame.
            title: Plot title.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cov_matrix, cmap="coolwarm", annot=False, square=True, cbar_kws={"label": "Covariance"})
        plt.title(title)
        plt.tight_layout()
        plt.show()


