"""
Tests for data cleaning functions: missing values, duplicates, timestamp parsing.
"""

import pandas as pd

from app.process_pipeline import clean_data


def test_clean_data_drops_missing_user_or_activity():
    """
    Rows with missing user_id or activity should be removed.
    """
    df = pd.DataFrame(
        {
            "user_id": ["u1", None, "u2"],
            "activity": ["login", "view_page", None],
            "timestamp": [
                "2025-01-01T10:00:00",
                "2025-01-01T10:05:00",
                "2025-01-01T10:10:00",
            ],
        }
    )

    cleaned = clean_data(df)

    # Only the first row has both user_id and activity
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["user_id"] == "u1"
    assert cleaned.iloc[0]["activity"] == "login"


def test_clean_data_removes_duplicates():
    """
    Exact duplicate rows should be removed from the cleaned dataset.
    """
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "activity": ["login", "login"],
            "timestamp": [
                "2025-01-01T10:00:00",
                "2025-01-01T10:00:00",
            ],
        }
    )
    cleaned = clean_data(df)

    # Duplicates should be collapsed into a single row
    assert len(cleaned) == 1


def test_clean_data_parses_timestamps():
    """
    Timestamps should be parsed into pandas datetime and invalid ones dropped.
    """
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "activity": ["login", "login"],
            "timestamp": ["2025-01-01T10:00:00", "not-a-date"],
        }
    )

    cleaned = clean_data(df)

    # Only the valid timestamp row should remain
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["user_id"] == "u1"
    # Ensure timestamp is a datetime type
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])
