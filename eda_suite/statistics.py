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

    def get_skewness(self) -> pd.DataFrame:
        """Returns skewness for each numerical column."""
        records = []
        for col in self._numerical_cols:
            val = stats.skew(self._df[col].dropna())
            records.append({"Feature": col, "Skewness": val})
        return pd.DataFrame.from_records(records)

    def get_normality(self) -> pd.DataFrame:
        """
        Runs D’Agostino–Pearson normality test on each numerical column.
        Returns a DataFrame of p-values and test statistics.
        """
        records = []
        for col in self._numerical_cols:
            k2, p = stats.normaltest(self._df[col].dropna())
            records.append({"Feature": col, "p_value": p, "k2": k2})
        return pd.DataFrame.from_records(records)
