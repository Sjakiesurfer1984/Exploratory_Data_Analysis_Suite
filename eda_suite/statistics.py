import pandas as pd

class StatisticsCalculator:
    """Performs statistical calculations on the DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe
        self._numerical_cols = self._df.select_dtypes(include=['number']).columns

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Calculates descriptive statistics for numerical columns."""
        return self._df[self._numerical_cols].describe()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculates the correlation matrix for numerical columns."""
        return self._df[self._numerical_cols].corr()