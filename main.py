from xgboost import data

from app.loader import load_model
from pathlib import Path
from app.utils import build_applicant_from_dict
from app.predict import two_stage_prediction

clf_path = Path("models/stage_1_classification_model.pkl")
reg_path = Path("models/stage_2_regression_model.pkl")

cls, reg = load_model(clf_path=clf_path, reg_path=reg_path)


def run_cli():
    data = {
        'no_of_dependents': 2,
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 1200000,
        'loan_amount': 1000000,
        'loan_term': 12,
        'cibil_score': 750,
        'residential_assets_value': 100000,
        'commercial_assets_value': 50000,
        'luxury_assets_value': 25000,
        'bank_asset_value': 15000
    }


    df = build_applicant_from_dict(data=data, expected_cols=list(cls.feature_names_in_))
    # print(df.head())
    print(two_stage_prediction(cls, reg, df))

if __name__ == "__main__":
    run_cli()