import pandas as pd
from src.data.loader import load_data

def test_load_data_existing_file(tmp_path):
    sample = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })
    file_path = tmp_path / "sample.csv"
    sample.to_csv(file_path, index=False)

    df = load_data(str(file_path))

    assert df is not None
    assert list(df.columns) == ["A", "B"]
    assert df.shape == (3, 2)


def test_load_data_missing_file():
    df = load_data("non_existing_file.csv")
    assert df is None
