import pandas as pd
from src.features.preprocessor import process_features


def test_process_features_preserves_columns():
    df = pd.DataFrame({
        "Gender": ["M", "F"],
        "Purchase_Category": ["Electronics", "Clothing"],
        "NumericFeature": [10, 20]
    })

    processed = process_features(df)

    assert list(processed.columns) == ["Gender", "Purchase_Category", "NumericFeature"]
    assert processed["Gender"].dtype.name == "category"
    assert processed["Purchase_Category"].dtype.name == "category"
    assert processed["NumericFeature"].dtype == df["NumericFeature"].dtype
