from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from steps.ingest import STROKE_CSV_FILE_FORMAT


def transformer_fn():
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, STROKE_CSV_FILE_FORMAT.numeric_features),
            ("cat", categorical_transformer, STROKE_CSV_FILE_FORMAT.categorical_features),
            ("bool", 'passthrough', [item for item in STROKE_CSV_FILE_FORMAT.boolean_features if item != 'stroke']),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])
