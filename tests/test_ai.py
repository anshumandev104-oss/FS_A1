"""
Tests for the AI component (KMeans clustering).
"""

import pandas as pd

from app.ai_model.kmeans_model import build_user_feature_matrix, cluster_users


def test_build_user_feature_matrix_shape():
    """
    The feature matrix should have one row per user and one column per activity.
    """
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "activity": ["login", "view_page", "login"],
            "timestamp": [
                "2025-01-01T10:00:00",
                "2025-01-01T10:05:00",
                "2025-01-01T10:10:00",
            ],
        }
    )
    features = build_user_feature_matrix(df)

    # Expect 2 users
    assert features["user_id"].nunique() == 2
    # Expect at least login and view_page columns
    assert "login" in features.columns
    assert "view_page" in features.columns


def test_cluster_users_cluster_count():
    """
    KMeans should produce the requested number of clusters (if enough users).
    """
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "activity": [
                "login",
                "view_page",
                "login",
                "logout",
            ],
            "timestamp": [
                "2025-01-01T10:00:00",
                "2025-01-01T10:05:00",
                "2025-01-01T10:10:00",
                "2025-01-01T10:15:00",
            ],
        }
    )

    clusters_df, model = cluster_users(df, n_clusters=2)

    # We requested 2 clusters, so model.n_clusters_ should be 2
    assert model.n_clusters == 2
    # Each user appears exactly once in the clusters mapping
    assert clusters_df["user_id"].nunique() == 3
