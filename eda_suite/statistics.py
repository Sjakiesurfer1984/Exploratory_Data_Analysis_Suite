from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage
from scipy import stats
import pandas as pd
import numpy as np

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

# ==============================================================================
# Additional statistical computations for PCA, clustering, and embeddings
# ==============================================================================
    
    def get_pca_scatter_data(
        self,
        columns: list[str],
        n_components: int = 2,
        scale: bool = True
    ) -> pd.DataFrame:
        """
        Compute PCA for 2D scatter visualisation.
        Returns a DataFrame containing the first two principal components.
    
        Args:
            columns (list[str]): Columns to include.
            n_components (int): Number of principal components to compute.
            scale (bool): Whether to standardise features before PCA.
    
        Returns:
            pd.DataFrame: PC scores (columns = PC1, PC2, ...)
        """
        numeric_df = self._df[columns].select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print("No numeric data available for PCA scatter.")
            return pd.DataFrame()
    
        X = numeric_df.values
        if scale:
            X = StandardScaler().fit_transform(X)
    
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X)
        pc_names = [f"PC{i+1}" for i in range(n_components)]
    
        return pd.DataFrame(pcs, columns=pc_names, index=numeric_df.index)
    
    
    # ----------------------------------------------------------------------
    def get_k_distance(
        self,
        columns: list[str],
        k: int = 4
    ) -> pd.DataFrame:
        """
        Compute the distance to each sample's k-th nearest neighbour.
        Used for DBSCAN elbow plot diagnostics.
    
        Args:
            columns (list[str]): Numerical columns to include.
            k (int): Neighbour rank (e.g., 4 for 4th nearest).
    
        Returns:
            pd.DataFrame: Sorted distances.
        """
        numeric_df = self._df[columns].select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print("No numeric data for k-distance calculation.")
            return pd.DataFrame()
    
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(numeric_df)
        distances, _ = neigh.kneighbors(numeric_df)
        k_distances = np.sort(distances[:, k - 1])
    
        return pd.DataFrame({"k_distance": k_distances})
    
    
    # ----------------------------------------------------------------------
    def get_hierarchical_linkage(
        self,
        columns: list[str],
        method: str = "ward"
    ) -> np.ndarray:
        """
        Compute the linkage matrix for hierarchical clustering dendrograms.
    
        Args:
            columns (list[str]): Numerical columns to include.
            method (str): Linkage method (ward, average, complete, single).
    
        Returns:
            np.ndarray: Linkage matrix.
        """
        numeric_df = self._df[columns].select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print("No numeric data for hierarchical clustering.")
            return np.array([])
    
        return linkage(numeric_df, method=method)
    
    
    # ----------------------------------------------------------------------
    def get_tsne_embedding(
        self,
        columns: list[str],
        perplexity: int = 30,
        n_components: int = 2,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute a t-SNE embedding for non-linear structure visualisation.
    
        Args:
            columns (list[str]): Numerical columns.
            perplexity (int): Balance between local/global structure.
            n_components (int): Usually 2 for visualisation.
            random_state (int): Reproducibility seed.
    
        Returns:
            pd.DataFrame: t-SNE coordinates.
        """
        numeric_df = self._df[columns].select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print("No numeric data for t-SNE embedding.")
            return pd.DataFrame()
    
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            learning_rate="auto",
            init="pca"
        )
        embedding = tsne.fit_transform(numeric_df.values)
        colnames = [f"Dim{i+1}" for i in range(n_components)]
        return pd.DataFrame(embedding, columns=colnames, index=numeric_df.index)
    
    
    # ----------------------------------------------------------------------
    def get_umap_embedding(
        self,
        columns: list[str],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute UMAP embedding for global structure visualisation.
    
        Args:
            columns (list[str]): Numerical columns to include.
            n_neighbors (int): Local neighbourhood size.
            min_dist (float): Controls tightness of clusters.
            random_state (int): Random seed.
    
        Returns:
            pd.DataFrame: UMAP embedding (2D).
        """
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Run: pip install umap-learn")
            return pd.DataFrame()
    
        numeric_df = self._df[columns].select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print("No numeric data for UMAP embedding.")
            return pd.DataFrame()
    
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        embedding = reducer.fit_transform(numeric_df.values)
        colnames = ["UMAP1", "UMAP2"]
        return pd.DataFrame(embedding, columns=colnames, index=numeric_df.index)
  
    # ----------------------------------------------------------------------
    # K-Means Elbow Curve
    # ----------------------------------------------------------------------  
    def compute_elbow_curve(
        self,
        X: np.ndarray,
        k_range: range = range(1, 11),
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute K-Means inertia (within-cluster sum of squares) for a range of k values.
    
        Args:
            X (np.ndarray): 2D array of scaled numerical features.
            k_range (range): Range of k (number of clusters) to evaluate.
            random_state (int): Seed for reproducibility.
    
        Returns:
            pd.DataFrame: DataFrame with columns ['k', 'inertia'].
        """
        results = []
        for k in k_range:
            kmeans = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
            kmeans.fit(X)
            results.append({"k": k, "inertia": kmeans.inertia_})
        
        return pd.DataFrame(results)

    # ----------------------------------------------------------------------
    # Automatic elbow point detection (2nd derivative)
    # ----------------------------------------------------------------------   
    def elbow_by_second_diff(self, ks: list, inertia: list) -> int:
        """
        Automatically estimate the optimal number of clusters (k) 
        using the 2nd derivative ('maximum curvature') method.
    
        Args:
            ks (list): List or range of tested k values.
            inertia (list): Corresponding inertia (within-cluster SSE) values.
    
        Returns:
            int: Estimated optimal k value.
        """
        ks = np.asarray(ks)
        I = np.asarray(inertia)
    
        # First and second discrete differences
        d1 = I[:-1] - I[1:]
        d2 = d1[:-1] - d1[1:]
    
        k_star = ks[1:-1][np.argmax(d2)]
        print(f"Detected elbow (max curvature) at k = {int(k_star)}")
        return int(k_star)

    def compute_silhouette_scores(self, X, k_range=range(2, 11), random_state=42):
        """
        Computes average silhouette score for different cluster counts.

        Args:
            X (np.ndarray): Scaled feature matrix.
            k_range (range): Range of k values to test.
            random_state (int): For reproducibility.

        Returns:
            pd.DataFrame: DataFrame with k and silhouette scores.
        """
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append({"k": k, "silhouette_score": score})
        return  pd.DataFrame(scores)

    def class_distribution(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Computes raw counts and proportions for a categorical column.
        Useful for identifying imbalanced categories during EDA.
        """
        counts = df[column].value_counts(dropna=False)
        proportions = counts / counts.sum()
        summary = pd.DataFrame({
            "Count": counts,
            "Proportion": proportions.round(4)
        })
        return summary


