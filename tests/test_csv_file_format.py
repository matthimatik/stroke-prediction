import pytest

from custom.csv_file_format import CSVFileFormat, FeatureDtype, get_features_of_type

def test_get_features_of_type():
    """Test the get_features_of_type function."""
    features = {
        "feature_cat": FeatureDtype.CATEGORY,
        "feature_bool": FeatureDtype.BOOL,
        "feature_float": FeatureDtype.FLOAT64,
    }
    assert get_features_of_type(features, FeatureDtype.CATEGORY) == ["feature_cat"]
    assert get_features_of_type(features, FeatureDtype.BOOL) == ["feature_bool"]
    assert get_features_of_type(features, FeatureDtype.FLOAT64) == ["feature_float"]


def test_csv_file_format():
    """Test the CSV file format."""
    features = {
        "feature_cat": FeatureDtype.CATEGORY,
        "feature_bool": FeatureDtype.BOOL,
        "feature_float": FeatureDtype.FLOAT64,
    }
    csv_file_format = CSVFileFormat(features)

    assert csv_file_format.categorical_features == ["feature_cat"]
    assert csv_file_format.boolean_features == ["feature_bool"]
    assert csv_file_format.numeric_features == ["feature_float"]
    assert csv_file_format.features_with_types_as_strings == {
        "feature_cat": "category",
        "feature_bool": "bool",
        "feature_float": "float64",
    }


if __name__ == "__main__":
    pytest.main()
