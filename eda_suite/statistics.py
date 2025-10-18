import pandas as pd
import scipy.stats as stats

class StatisticsCalculator:
    """Performs statistical calculations on the DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe
        self._numerical_cols = self._df.select_dtypes(include=['number']).columns

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Calculates descriptive statistics for numerical columns."""
        return self._df[self._numerical_cols].describe()

    def get_correlation_matrix(self, columns: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
        """
        Returns the correlation matrix over numerical columns (default: Pearson).
        method can be 'pearson', 'spearman', or 'kendall'.
        """
        cols = columns or self._numerical_cols
        # Guard against empty selection
        if len(cols) == 0:
            return pd.DataFrame()
        return self._df[cols].corr(method=method)

    def get_skewness(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Returns skewness for each specified numerical column."""
        cols = columns or self._numerical_cols
        records = []
        for col in cols:
            val = stats.skew(self._df[col].dropna())
            records.append({"Feature": col, "Skewness": val})
        return pd.DataFrame.from_records(records)

    def get_normality(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Runs D’Agostino–Pearson normality test on each specified numerical column."""
        cols = columns or self._numerical_cols
        records = []
        for col in cols:
            k2, p = stats.normaltest(self._df[col].dropna())
            records.append({"Feature": col, "p_value": p, "k2": k2})
        return pd.DataFrame.from_records(records)

    def compute_covariance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the covariance matrix for numerical columns.

        Args:
            df: The DataFrame containing numerical features.

        Returns:
            DataFrame: Covariance matrix.
        """
        numeric_df = df.select_dtypes(include="number")
        cov_matrix = numeric_df.cov()
        return cov_matrix

    def summarize_covariance(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics of covariance magnitudes.

        Returns:
            Summary DataFrame with mean, median, and std of off-diagonal covariances.
        """
        import numpy as np
        # Extract off-diagonal values only
        off_diag = cov_matrix.values[np.triu_indices_from(cov_matrix, k=1)]
        summary = {
            "mean_covariance": off_diag.mean(),
            "median_covariance": np.median(off_diag),
            "std_covariance": off_diag.std()
        }
        return pd.DataFrame([summary])
