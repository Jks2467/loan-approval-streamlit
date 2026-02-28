

def two_stage_prediction(cls, reg, application_df):

    proba = cls.predict_proba(application_df)

    preds = cls.predict(application_df)

    result = []

    # array indexes for probability and prediction
    array_idx = 0
    approval_idx = 1
    

    approval = int(preds[array_idx])
    approval_proba = float(proba[array_idx][approval_idx])

    reg_pred = None

    if approval == 1:
        application_df_reg = application_df.copy()
        application_df_reg['loan_status'] = "Approved"
        reg_pred = float(reg.predict(application_df_reg)[0])

    result.append(
        {
            "approval": approval,
            "approval_proba": approval_proba,
            "regression_prediction": reg_pred
        }
    )

    
    return result