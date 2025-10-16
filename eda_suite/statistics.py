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

    # ------------------------------------------------------------
    # Normality and Skewness Tests
    # ------------------------------------------------------------

    def _check_normality(self, col: str) -> None:
        """Runs a D’Agostino–Pearson normality test on a single column."""
        k2, p = stats.normaltest(self._df[col])
        print(f"{col}: p-value = {p:.4f}")
        if p > 0.05:
            print("  → Approximately normal.")
        else:
            print("  → Deviates from normality.")

    def _check_skewness(self, col: str) -> None:
        """Computes skewness for a single column and interprets it."""
        skew_val = stats.skew(self._df[col])
        print(f"{col}: skewness = {skew_val:.3f}")
        if abs(skew_val) < 0.5:
            print("  → Fairly symmetric.")
        elif abs(skew_val) < 1:
            print("  → Moderately skewed.")
        else:
            print("  → Highly skewed – consider log or power transform.")
