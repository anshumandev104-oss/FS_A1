"""
End-to-end data processing pipeline for activity logs.

Responsibilities:
- Ingest CSV (local file or from S3 when run in Lambda)
- Clean the data (missing values, timestamp parsing, duplicate removal)
- Run basic analytics (total users, most common activity, most active user)
- Apply AI model (KMeans clustering of user behavior)
- Save cleaned data and analytics to CSV (locally or back to S3 in Lambda)

This script is written to:
- Be runnable locally for development and testing
- Be used as an AWS Lambda handler for S3-triggered processing
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, Tuple

import boto3
import pandas as pd

from app.ai_model.kmeans_model import cluster_users


def load_csv_from_local(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the local filesystem.

    Args:
        path: Path to the input CSV file.

    Returns:
        DataFrame with the raw activity logs.
    """
    return pd.read_csv(path)


def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """
    Load a CSV file from S3 into a pandas DataFrame.

    This is used when the pipeline runs inside AWS Lambda.

    Args:
        bucket: Name of the S3 bucket.
        key: Key (path) of the CSV object inside the bucket.

    Returns:
        DataFrame with the raw activity logs.
    """
    s3_client = boto3.client("s3")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    # Read the object body into memory and decode to text
    data_bytes = obj["Body"].read()
    # Use io.StringIO so pandas can read it like a file
    csv_buffer = io.StringIO(data_bytes.decode("utf-8"))
    df = pd.read_csv(csv_buffer)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw activity logs DataFrame.

    Steps:
    - Drop rows where user_id or activity is missing (cannot use these logs)
    - Parse timestamp column into datetime
    - Drop rows with invalid timestamps
    - Remove exact duplicate rows

    Args:
        df: Raw DataFrame with columns ['user_id', 'activity', 'timestamp'].

    Returns:
        Cleaned DataFrame ready for analytics and modeling.
    """
    # Work on a copy to avoid modifying the original DataFrame in-place
    cleaned = df.copy()

    # Standardize column names if needed
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    # Drop rows where user_id or activity is missing (core identifiers)
    cleaned = cleaned.dropna(subset=["user_id", "activity"])

    # Parse timestamps; coerce errors so invalid strings become NaT
    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"], errors="coerce"
    )

    # Drop rows where timestamp could not be parsed
    cleaned = cleaned.dropna(subset=["timestamp"])

    # Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    return cleaned


def run_basic_analytics(cleaned: pd.DataFrame) -> Dict[str, str]:
    """
    Compute basic analytics on the cleaned data.

    Metrics:
    - total_users: number of unique users
    - most_common_activity: activity value with the highest count
    - most_active_user: user_id with the highest number of rows

    Args:
        cleaned: Cleaned DataFrame.

    Returns:
        Dictionary with simple analytics results.
    """
    total_users = cleaned["user_id"].nunique()

    # Activity with the highest frequency
    most_common_activity = (
        cleaned["activity"].value_counts().idxmax()
        if not cleaned.empty
        else None
    )

    # User with the highest number of actions
    most_active_user = (
        cleaned["user_id"].value_counts().idxmax()
        if not cleaned.empty
        else None
    )

    # Convert values to strings to make serialization easy
    analytics = {
        "total_users": str(total_users),
        "most_common_activity": (
            str(most_common_activity)
            if most_common_activity is not None
            else ""
        ),
        "most_active_user": (
            str(most_active_user) if most_active_user is not None else ""
        ),
    }

    return analytics


def apply_ai_model(cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the AI model (KMeans clustering) to the cleaned data.

    This function:
    - Clusters users by their behavior
    - Returns a DataFrame mapping user_id to cluster labels

    Args:
        cleaned: Cleaned DataFrame.

    Returns:
        DataFrame with ['user_id', 'cluster'].
    """
    user_clusters, _ = cluster_users(cleaned_df=cleaned, n_clusters=3)
    return user_clusters


def save_outputs_local(
    cleaned: pd.DataFrame,
    analytics: Dict[str, str],
    output_dir: str = "output",
) -> Tuple[str, str]:
    """
    Save cleaned data and analytics locally in CSV/JSON format.

    Args:
        cleaned: Cleaned DataFrame.
        analytics: Dictionary with analytics metrics.
        output_dir: Directory where outputs will be written.

    Returns:
        Tuple of paths (cleaned_csv_path, analytics_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    cleaned_path = os.path.join(output_dir, "cleaned_data.csv")
    analytics_path = os.path.join(output_dir, "analytics_summary.json")

    cleaned.to_csv(cleaned_path, index=False)

    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)

    return cleaned_path, analytics_path


def save_outputs_to_s3(
    cleaned: pd.DataFrame,
    analytics: Dict[str, str],
    bucket: str,
    prefix: str = "results/",
    original_key: str | None = None,
) -> Tuple[str, str]:
    """
    Save cleaned data and analytics to S3.

    File naming convention:
    - cleaned data: prefix + 'cleaned_data_<original_filename>.csv'
    - analytics: prefix + 'analytics_summary_<original_filename>.json'

    Args:
        cleaned: Cleaned DataFrame.
        analytics: Dictionary with analytics metrics.
        bucket: Target S3 bucket name.
        prefix: Folder/prefix inside the bucket for outputs.
        original_key: Original S3 key of the input file (for naming).

    Returns:
        Tuple of S3 keys (cleaned_key, analytics_key).
    """
    s3_client = boto3.client("s3")

    # Derive base name from original_key if provided
    if original_key:
        base_name = os.path.splitext(os.path.basename(original_key))[0]
    else:
        base_name = "input"

    cleaned_key = f"{prefix}cleaned_data_{base_name}.csv"
    analytics_key = f"{prefix}analytics_summary_{base_name}.json"

    # Write cleaned CSV to an in-memory buffer
    csv_buffer = io.StringIO()
    cleaned.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=bucket, Key=cleaned_key, Body=csv_buffer.getvalue()
    )

    # Write analytics JSON to an in-memory buffer
    json_buffer = io.StringIO()
    json.dump(analytics, json_buffer)
    s3_client.put_object(
        Bucket=bucket, Key=analytics_key, Body=json_buffer.getvalue()
    )

    return cleaned_key, analytics_key


def run_pipeline_on_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Run the full pipeline on an in-memory DataFrame.

    This is used for both local execution and Lambda.

    Steps:
    - Clean data
    - Compute analytics
    - Apply AI model and merge cluster labels into cleaned data

    Args:
        df: Raw input DataFrame.

    Returns:
        Tuple of:
        - cleaned DataFrame with cluster column
        - analytics dictionary
    """
    cleaned = clean_data(df)
    analytics = run_basic_analytics(cleaned)

    # Apply AI model and merge cluster labels into cleaned data
    clusters_df = apply_ai_model(cleaned)
    cleaned_with_clusters = cleaned.merge(
        clusters_df, on="user_id", how="left"
    )

    return cleaned_with_clusters, analytics


def main_local(input_csv_path: str = "raw_logs_sample.csv") -> None:
    """
    Entry point for running the pipeline locally on your laptop.

    Reads the input CSV from the local path, runs the pipeline,
    and writes results to an 'output' directory.
    """
    raw_df = load_csv_from_local(input_csv_path)
    cleaned_with_clusters, analytics = run_pipeline_on_dataframe(raw_df)
    cleaned_path, analytics_path = save_outputs_local(
        cleaned_with_clusters, analytics
    )
    print(f"Saved cleaned data to: {cleaned_path}")
    print(f"Saved analytics summary to: {analytics_path}")


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    This is triggered by S3 when a new raw CSV is uploaded.

    Expected S3 event structure:
    - event['Records'][0]['s3']['bucket']['name'] -> bucket name
    - event['Records'][0]['s3']['object']['key'] -> key of uploaded CSV

    The handler:
    - Reads the CSV from S3
    - Runs the pipeline
    - Writes cleaned data and analytics back to S3 under 'results/' prefix
    """
    # Extract bucket and key from the S3 event notification
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    # Load the CSV from S3
    raw_df = load_csv_from_s3(bucket=bucket, key=key)

    # Run the pipeline
    cleaned_with_clusters, analytics = run_pipeline_on_dataframe(raw_df)

    # Save outputs back to S3
    cleaned_key, analytics_key = save_outputs_to_s3(
        cleaned_with_clusters,
        analytics,
        bucket=bucket,
        prefix="results/",
        original_key=key,
    )

    # Return a simple response for logging/monitoring
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Pipeline completed successfully",
                "cleaned_key": cleaned_key,
                "analytics_key": analytics_key,
            }
        ),
    }


if __name__ == "__main__":
    # Allow easy local testing: `python -m app.process_pipeline`
    main_local()
