from enum import Enum


class FeatureDtype(Enum):
    """The types of features."""
    CATEGORY = "category"
    BOOL = "bool"
    FLOAT64 = "float64"


FeatureName = str
Features = dict[FeatureName, FeatureDtype]
"""Feature names and their data types."""


def get_features_of_type(dtypes: Features, feature_dtype: FeatureDtype) -> list:
    """Get a list of features of the specified type."""
    return [feature for feature, dtype in dtypes.items() if dtype == feature_dtype]


class CSVFileFormat:
    """Represents the CSV file format."""
    features: Features

    def __init__(self, features: Features) -> None:
        self.features = features

    @property
    def categorical_features(self) -> list[str]:
        """Get a list of the categorical features."""
        return get_features_of_type(self.features, FeatureDtype.CATEGORY)

    @property
    def boolean_features(self) -> list[str]:
        """Get a list of the boolean features."""
        return get_features_of_type(self.features, FeatureDtype.BOOL)
    
    @property
    def numeric_features(self) -> list[str]:
        """Get a list of the numeric features."""
        return get_features_of_type(self.features, FeatureDtype.FLOAT64)

    @property
    def features_with_types_as_strings(self) -> dict[str, str]:
        """Get a dictionary of the features and their types as strings."""
        return {feature: dtype.value for feature, dtype in self.features.items()}
