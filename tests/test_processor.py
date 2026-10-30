import pytest
import pandas as pd

# Assuming the processor module is located at src/processor.py.
# Adjust the import path according to the actual project structure.
from src.processor import clean_text, batch_data


def test_clean_text_basic_operations():
    """Test that clean_text performs expected normalisation steps."""
    raw = "  This is a TEST! Check out https://example.com @user   "
    cleaned = clean_text(raw)

    # Expected transformations:
    # - strip leading/trailing whitespace
    # - lower‑casing
    # - removal of URLs
    # - removal of mentions (words starting with @)
    # - collapse multiple spaces into a single space
    assert cleaned == "this is a test! check out"
    # Ensure original string is not modified (function should be pure)
    assert raw == "  This is a TEST! Check out https://example.com @user   "


def test_clean_text_handles_none_and_empty():
    """clean_text should gracefully handle None or empty strings."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    # Whitespace only should become empty after stripping
    assert clean_text("   \t\n") == ""


def test_clean_text_preserves_non_ascii_characters():
    """Non‑ASCII characters (e.g., emojis, accented letters) should be kept."""
    raw = "¡Hola! 😊 https://test.com"
    cleaned = clean_text(raw)
    assert cleaned == "¡hola! 😊"


def test_batch_data_correct_batching():
    """batch_data should split a DataFrame into batches of the requested size."""
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(23)],
            "label": [i % 2 for i in range(23)],
        }
    )
    batch_size = 5
    batches = batch_data(df, batch_size)

    # Number of batches should be ceil(23 / 5) = 5
    assert len(batches) == 5

    # All batches except possibly the last must have exactly batch_size rows
    for batch in batches[:-1]:
        assert len(batch) == batch_size

    # The last batch should contain the remainder
    assert len(batches[-1]) == 23 % batch_size

    # Re‑concatenating the batches should reproduce the original DataFrame (order preserved)
    reconstructed = pd.concat(batches, ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, df.reset_index(drop=True))


def test_batch_data_invalid_batch_size():
    """batch_data should raise a ValueError for non‑positive batch sizes."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        batch_data(df, 0)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        batch_data(df, -5)


def test_batch_data_preserves_dataframe_index_and_columns():
    """The returned batches must retain original column names and index types."""
    df = pd.DataFrame(
        {
            "id": range(10),
            "text": [f"msg {i}" for i in range(10)],
        }
    ).set_index("id")
    batch_size = 3
    batches = batch_data(df, batch_size)

    # Each batch should keep the same column set
    for batch in batches:
        assert list(batch.columns) == ["text"]
        # Index type should be the same as the original (Int64Index)
        assert isinstance(batch.index, pd.Int64Index)

    # Verify that the union of batch indices equals the original index set
    combined_index = pd.Index([]).union_many([b.index for b in batches])
    pd.testing.assert_index_equal(combined_index.sort_values(), df.index.sort_values())
