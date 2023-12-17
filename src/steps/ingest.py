import pandas as pd



def load_stroke_dataset_csv_as_dataframe(location: str, file_format: str) -> pd.DataFrame:
    """Load content from the specified stroke dataset csv file as a Pandas DataFrame."""
    df = pd.read_csv(
        location,
        dtype={
            "gender": "category",
            "age": "float64",
            "hypertension": bool,
            "heart_disease": bool,
            "ever_married": bool,
            "work_type": "category",
            "Residence_type": "category",
            "avg_glucose_level": "float64",
            "bmi": "float64",
            "smoking_status": "category",
            "stroke": bool,
        },
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
    df = load_stroke_dataset_csv_as_dataframe(location, file_format)

    # Convert categorical columns to strings while keeping NaN values unchanged
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(str)

    return df