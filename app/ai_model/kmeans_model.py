"""
Simple KMeans clustering model for user activity behavior.

This module defines a function that:
- Builds a feature matrix from cleaned activity logs
- Applies KMeans clustering
- Returns a DataFrame with user_id and assigned cluster labels

Designed to be called from process_pipeline.py after data cleaning.
"""

from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans


def build_user_feature_matrix(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-level feature matrix from cleaned activity logs.

    Each row represents a user and each column is a count of a specific activity.

    Args:
        cleaned_df: DataFrame with at least columns ['user_id', 'activity'].

    Returns:
        DataFrame indexed by user_id with one column per activity type.
    """
    # Use a pivot table to count how many times each user performs each activity
    activity_counts = (
        cleaned_df.groupby(["user_id", "activity"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure user_id is a regular column rather than index for easier merging later
    activity_counts = activity_counts.reset_index()

    return activity_counts


def cluster_users(
    cleaned_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster users based on their activity behavior using KMeans.

    Args:
        cleaned_df: Cleaned activity logs with columns ['user_id', 'activity'].
        n_clusters: Number of clusters to create.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of:
        - DataFrame with columns ['user_id', 'cluster']
        - Fitted KMeans model instance
    """
    features_df = build_user_feature_matrix(cleaned_df)

    # Separate user_id and feature columns
    user_ids = features_df["user_id"]
    feature_columns = [col for col in features_df.columns if col != "user_id"]
    feature_matrix = features_df[feature_columns]

    # Instantiate and fit the KMeans model
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",  # use sensible default initializations
    )
    model.fit(feature_matrix)

    # Assign each user to a cluster
    cluster_labels = model.predict(feature_matrix)

    # Build a result DataFrame mapping user_id to cluster label
    user_clusters = pd.DataFrame(
        {
            "user_id": user_ids,
            "cluster": cluster_labels,
        }
    )

    return user_clusters, model
