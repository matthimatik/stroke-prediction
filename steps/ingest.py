import pandas as pd

from custom.csv_file_format import CSVFileFormat, FeatureDtype

STROKE_CSV_FILE_FORMAT = CSVFileFormat({
    "gender": FeatureDtype.CATEGORY,
    "age": FeatureDtype.FLOAT64,
    "hypertension": FeatureDtype.BOOL,
    "heart_disease": FeatureDtype.BOOL,
    "ever_married": FeatureDtype.BOOL,
    "work_type": FeatureDtype.CATEGORY,
    "Residence_type": FeatureDtype.CATEGORY,
    "avg_glucose_level": FeatureDtype.FLOAT64,
    "bmi": FeatureDtype.FLOAT64,
    "smoking_status": FeatureDtype.CATEGORY,
    "stroke": FeatureDtype.BOOL,
})


def load_stroke_dataset_as_dataframe(location: str, file_format: str = "csv") -> pd.DataFrame:
    """Load content from the specified stroke dataset csv file as a Pandas DataFrame."""
    assert file_format == "csv", f"Unsupported file format: {file_format}. Currently only csv is supported."

    df = pd.read_csv(
        location,
        dtype=STROKE_CSV_FILE_FORMAT.features_with_types_as_strings,
        true_values=["Yes"],
        false_values=["No"],
        na_values=["Unknown"],
    )
    df.drop(["id"], axis=1, inplace=True)

    return df


def load_and_convert_stroke_dataset(location: str, file_format: str) -> pd.DataFrame:
    """
    Load stroke dataset csv file as a Pandas DataFrame and convert categorical columns to strings.
    Keeping NaN values unchanged.
    """
    df = load_stroke_dataset_as_dataframe(location, file_format)

    # Convert categorical columns to strings. TODO: keep NaN values unchanged
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)

    return df
