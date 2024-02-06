"""
Custom functions for the project
"""

# Basic Libraries & data manipulation
import os
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd
import woodwork as ww

# Data Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Feature Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce # pip install category_encoders

# Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Feature Engineering
import tsfresh
import featuretools
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Modelling & Evaluation
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




# Helper function for splitting the data into train and test sets

def split_by_id(df: pd.DataFrame, test_percent: float=0.2, random_state: int=123, plot: bool=True) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataframe into train and test, ensuring that no Consumer_number is present in both sets.

    This function splits the dataset into training and testing, with different consumers in each set, while
    maintaining the proportion of the target variable. It takes into account the Consumer_type variable, so that
    the proportion of each Consumer_type is the same in both sets (aprox).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    test_percent : float, optional
        The percentage of the dataset to be used for testing. The default is 0.2.
    random_state : int, optional
        The random state to be used for reproducibility. The default is 123.
    plot : bool, optional
        Whether to plot the distribution of the Consumer_type in the train and test sets. The default is True.

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The testing set.
    """
    id_by_type = df.groupby('Consumer_type')['Consumer_number'].unique().to_dict()
    df_id_by_type = pd.DataFrame(list(id_by_type.items()), columns=['Consumer_type', 'Consumer_number'])
    df_id_by_type = df_id_by_type.explode('Consumer_number').reset_index(drop=True)
    df_id_by_type.head()

    # Split the Consumer_number into train and test while mantaining the proportion of the Consumer_type
    train_consumers, test_consumers = train_test_split(df_id_by_type, test_size=test_percent, random_state=random_state, stratify=df_id_by_type['Consumer_type'])

    train = df[df['Consumer_number'].isin(train_consumers['Consumer_number'])]
    test = df[df['Consumer_number'].isin(test_consumers['Consumer_number'])]


    if plot:
        # Plot the distribution of the Consumer_type in the train and test sets
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=train.drop_duplicates(subset=['Consumer_number'])['Consumer_type'], name='Train'))
        fig.add_trace(go.Histogram(x=test.drop_duplicates(subset=['Consumer_number'])['Consumer_type'],  name='Test'))
        fig.update_layout(
            title='Distribution of the Consumer_type in the train and test sets',
            xaxis_title='Consumer_type',
            yaxis_title='Number of Consumers',
            barmode='overlay',
        )
        # Apply styling to the plot to be more professional, paper-like
        # choose the figure font
        font_dict=dict(
                    size=12,
                    color='black'
                    )
        # general figure formatting
        fig.update_layout(font=font_dict,  # font formatting
                        plot_bgcolor='white',  # background color
                        width=850,  # figure width
                        height=700,  # figure height
                        margin=dict(r=10,t=50,b=10)  # remove white space 
                        )
        # x and y-axis formatting
        fig.update_yaxes(showline=True,  # add line at x=0
                        linecolor='black',  # line color
                        linewidth=2.4, # line size
                        ticks='outside',  # ticks outside axis
                        tickfont=font_dict, # tick label font
                        mirror='allticks',  # add ticks to top/right axes
                        tickwidth=2.4,  # tick width
                        tickcolor='black',  # tick color
                        )
        fig.update_xaxes(showline=True,
                        showticklabels=True,
                        linecolor='black',
                        linewidth=2.4,
                        ticks='outside',
                        tickfont=font_dict,
                        mirror='allticks',
                        tickwidth=2.4,
                        tickcolor='black',
                        )
        fig.show()

    return train, test


def subsample_dataset(X, y, cluster_count=2, sample_size=7000, random_state=123):
    """
    Subsamples the dataset by selecting representative samples from each cluster.
    
    This function applies PCA to reduce the dimensionality of the dataset, then applies K-means to identify clusters,
    and finally selects representative samples from each cluster.
    
    Parameters
    ----------
    X : pd.DataFrame
        The features of the dataset.
    y : pd.Series
        The target variable of the dataset.
    cluster_count : int, optional
        The number of clusters to be used. The default is 50.
    sample_size : int, optional
        The number of samples to be selected from each cluster. The default is 300.
    random_state : int, optional
        The random seed to be used for reproducibility. The default is 123.
    
    Returns
    -------
    X_subsampled : pd.DataFrame
        The subsampled features.
    y_subsampled : pd.Series
        The subsampled target variable.
    subsampled_indices : list
        The indices of the subsampled dataset.
    
    """
    # Apply PCA to reduce dimensionality (optional, depending on the dimensionality of your data)
    pca = PCA(n_components=0.95, random_state=random_state)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X)

    # Apply K-means to identify clusters
    kmeans = KMeans(n_clusters=cluster_count, random_state=random_state)  # Adjust the number of clusters as needed
    clusters = kmeans.fit_predict(X_pca)

    # Subsample: select representative samples from each cluster
    subsampled_indices = []
    for cluster in np.unique(clusters):
        # Find indices of samples in this cluster
        indices_in_cluster = np.where(clusters == cluster)[0]
        # Randomly select n samples from this cluster
        selected_indices = np.random.choice(indices_in_cluster, size=sample_size, replace=False)  # Adjust 'size' as needed
        subsampled_indices.extend(selected_indices)

    # Create the new subsampled dataset
    X_subsampled = X.iloc[subsampled_indices]
    y_subsampled = y.iloc[subsampled_indices]

    return X_subsampled, y_subsampled, subsampled_indices


class OptimalClusterFinder:
    """
    Finds the optimal number of clusters for K-means clustering using the elbow method, silhouette method,
    Calinski-Harabasz method, and Davies-Bouldin method.
    
    Parameters
    ----------
    max_clusters : int, optional
        The maximum number of clusters to be tested. The default is 10.
    
    Attributes
    ----------
    scores : dict
        The optimal number of clusters for each method.

    Examples
    --------
    >>> from functions import OptimalClusterFinder
    >>> finder = OptimalClusterFinder(max_clusters=10)
    >>> optimal_clusters = finder.fit(X)
    >>> finder.scores
    {'elbow': 3, 'silhouette': 3, 'calinski_harabasz': 3, 'davies_bouldin': 3}

    """
    def __init__(self, max_clusters=10):
        self.max_clusters = max_clusters
        self.scores = {}

    def _elbow_method(self, X):
        """
        Finds the optimal number of clusters using the elbow method.

        The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis
        designed to help finding the appropriate number of clusters in a dataset. The optimal number of clusters is
        usually defined as the point at which the within-cluster sum of squares (WCSS) becomes inflexible.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to be clustered.

        Returns
        -------
        optimal_clusters : int
            The optimal number of clusters.

        """
        print('Finding the optimal number of clusters using the elbow method...')
        sse = {}
        for k in tqdm(range(1, self.max_clusters)):
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(X)
            sse[k] = kmeans.inertia_
        elbow_point = np.diff(list(sse.values()))
        optimal_clusters = np.argmin(elbow_point) + 2  # +2 as the diff is between subsequent clusters
        return optimal_clusters

    def _silhouette_method(self, X):
        """
        Finds the optimal number of clusters using the silhouette method.

        The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other
        clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is
        well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value,
        then the clustering configuration is appropriate. If many points have a low or negative value, then the
        clustering configuration may have too many or too few clusters.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to be clustered.

        Returns
        -------
        optimal_clusters : int
            The optimal number of clusters.

        """
        print('Finding the optimal number of clusters using the silhouette method...')
        silhouette_avg = {}
        for k in tqdm(range(2, self.max_clusters)):
            kmeans = KMeans(n_clusters=k, random_state=1)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg[k] = silhouette_score(X, cluster_labels)
        optimal_clusters = max(silhouette_avg, key=silhouette_avg.get)
        return optimal_clusters

    def _calinski_harabasz_method(self, X):
        """
        Finds the optimal number of clusters using the Calinski-Harabasz method.

        The Calinski-Harabasz index is the ratio of the sum of between-clusters dispersion and of inter-cluster
        dispersion for all clusters (where dispersion is defined as the sum of distances squared). The score is higher
        when clusters are dense and well separated, which relates to a standard concept of a cluster.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to be clustered.

        Returns
        -------
        optimal_clusters : int
            The optimal number of clusters.

        """
        print('Finding the optimal number of clusters using the Calinski-Harabasz method...')
        calinski_harabasz_scores = {}
        for k in tqdm(range(2, self.max_clusters)):
            kmeans = KMeans(n_clusters=k, random_state=1)
            cluster_labels = kmeans.fit_predict(X)
            calinski_harabasz_scores[k] = calinski_harabasz_score(X, cluster_labels)
        optimal_clusters = max(calinski_harabasz_scores, key=calinski_harabasz_scores.get)
        return optimal_clusters

    def _davies_bouldin_method(self, X):
        """
        Finds the optimal number of clusters using the Davies-Bouldin method.

        The Davies-Bouldin index is the average similarity measure of each cluster with its most similar cluster,
        where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which
        are farther apart and less dispersed will result in a better score.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to be clustered.

        Returns
        -------
        optimal_clusters : int
            The optimal number of clusters.

        """
        print('Finding the optimal number of clusters using the Davies-Bouldin method...')
        davies_bouldin_scores = {}
        for k in tqdm(range(2, self.max_clusters)):
            kmeans = KMeans(n_clusters=k, random_state=1)
            cluster_labels = kmeans.fit_predict(X)
            davies_bouldin_scores[k] = davies_bouldin_score(X, cluster_labels)
        optimal_clusters = min(davies_bouldin_scores, key=davies_bouldin_scores.get)
        return optimal_clusters

    def fit(self, X):
        self.scores['elbow'] = self._elbow_method(X)
        # self.scores['silhouette'] = self._silhouette_method(X)
        self.scores['calinski_harabasz'] = self._calinski_harabasz_method(X)
        self.scores['davies_bouldin'] = self._davies_bouldin_method(X)
        return self.scores
    

def plot_confusion_matrix(matrix, class_names):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.set(font_scale=0.8)
    sns.heatmap(matrix, annot=True, annot_kws={'size':8},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = class_names
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()

    return fig


def generate_synthetic_samples(df, minority_class, n_samples):
    """
    Generates synthetic samples for the minority class in a DataFrame.
    
    It calculates the mean and standard deviation of the minority class, and then takes samples from
    a Gaussian distribution with these parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    minority_class : str
        The name of the minority class.
    n_samples : int
        The number of synthetic samples to generate.
        
    Returns
    -------
    df_synthetic : pd.DataFrame
        The synthetic samples.
        
    """
    df_minority = df[df['Consumer_type'] == minority_class]

    # Drop the Consumer_number and Consumer_type columns
    df_minority = df_minority.drop(columns=['Consumer_number', 'Consumer_type'])
    columns = df_minority.columns
    
    # Calculate the mean and standard deviation of the minority class
    mean = df_minority.mean()
    std = df_minority.std()/6

    # Generate synthetic samples
    synthetic_samples = []
    for i in range(n_samples):
        sample = np.random.normal(mean, std)
        synthetic_samples.append(sample)

    # Create a DataFrame with the synthetic samples
    df_synthetic = pd.DataFrame(synthetic_samples, columns=columns)
    df_synthetic['Consumer_number'] = 'synthetic'
    df_synthetic['Consumer_type'] = minority_class

    return df_synthetic


