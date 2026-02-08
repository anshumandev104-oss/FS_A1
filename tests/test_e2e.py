"""
End-to-end test: given input CSV, assert that output files are created.
This runs the pipeline locally (not in AWS).
"""

import os

import pandas as pd

from app.process_pipeline import main_local


def test_end_to_end_creates_output(tmp_path, monkeypatch):
    """
    Run the local pipeline on a temporary CSV and verify outputs exist.
    """
    # Create a small temporary input CSV
    input_csv = tmp_path / "raw_logs.csv"
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1"],
            "activity": ["login", "view_page", "logout"],
            "timestamp": [
                "2025-01-01T10:00:00",
                "2025-01-01T10:05:00",
                "2025-01-01T10:10:00",
            ],
        }
    )
    df.to_csv(input_csv, index=False)

    # Use a temporary output directory
    output_dir = tmp_path / "output"
    monkeypatch.chdir(tmp_path)

    # Run the local pipeline
    main_local(input_csv_path=str(input_csv))

    # Check that cleaned_data.csv and analytics_summary.json were created
    cleaned_path = output_dir / "cleaned_data.csv"
    analytics_path = output_dir / "analytics_summary.json"

    assert cleaned_path.exists()
    assert analytics_path.exists()
