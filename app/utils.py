import pandas as pd

def build_applicant_from_dict(data: dict, expected_cols: list) -> pd.DataFrame:

    df = pd.DataFrame([data])

    # clean white spaces for object type column values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    #check missing columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    return df[expected_cols]