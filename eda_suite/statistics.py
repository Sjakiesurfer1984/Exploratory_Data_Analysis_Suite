import pandas as pd
import numpy as np
from typing import Optional, List
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    def get_covariance_matrix(self, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Returns the covariance matrix for numerical columns.
        If 'columns' is provided, restrict to that subset.
        """
        # Filter for numeric dtypes
        numeric_df = self._df.select_dtypes(include=["number"])
    
        # If user passes columns, subset intersection
        if columns is not None:
            valid_cols = [c for c in columns if c in numeric_df.columns]
            numeric_df = numeric_df[valid_cols]
    
        # Return empty if nothing left
        if numeric_df.empty:
            return pd.DataFrame()
    
        return numeric_df.cov()

    def summarize_covariance(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise covariance magnitudes (mean, median, std of off-diagonals).
        """
        import numpy as np
        if cov_matrix.empty:
            return pd.DataFrame([{"mean_covariance": np.nan, "median_covariance": np.nan, "std_covariance": np.nan}])
        off_diag = cov_matrix.values[np.triu_indices_from(cov_matrix, k=1)]
        return pd.DataFrame([{
            "mean_covariance": off_diag.mean(),
            "median_covariance": np.median(off_diag),
            "std_covariance": off_diag.std()
        }])

    def get_pca_components(
        self,
        columns: list[str] | None = None,
        n_components: int | None = None,
        scale: bool = True
    ) -> pd.DataFrame:
        """
        Perform PCA and return explained variance ratios.

        Args:
            columns (list[str] | None): Columns to include. Defaults to all numeric.
            n_components (int | None): Number of components to compute.
            scale (bool): Whether to standardise data before PCA.

        Returns:
            pd.DataFrame: Table with PC index, explained variance, and cumulative ratio.
        """
        numeric_df = self._df.select_dtypes(include=["number"])
        if columns is not None:
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            print("No numerical columns available for PCA.")
            return pd.DataFrame()

        X = numeric_df.values
        if scale:
            X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=n_components)
        pca.fit(X)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        df_pca = pd.DataFrame({
            "Component": np.arange(1, len(explained_var) + 1),
            "ExplainedVarianceRatio": explained_var,
            "CumulativeVariance": cumulative_var
        })

        return df_pca
