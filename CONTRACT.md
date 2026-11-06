# Internal implementation contract

- Root: repository directory; Python 3.12; seed 42.
- Target in normalized frame: `default`; original XLS header: `default payment next month`.
- Features: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2..PAY_6, BILL_AMT1..6, PAY_AMT1..6. ID excluded.
- Split before EDA: stratified test 20%, then validation 25% of remaining; train/validation/test = 18000/6000/6000. sklearn train_test_split, random_state=42 each step. Membership: data/generated/splits.csv (row_position, split), ignored by Git.
- EDA interface: `src.eda.run_eda(df, train, root)`; frames have canonical original column names plus `default`; train contains only training rows; descriptive full-data quality permitted. EDA owns src/eda.py and images/eda_*, exports/eda_*, report/eda_notes.md.
- Model: logistic regression C=1, max_iter=3000, lbfgs, no class weighting; numeric imputer+StandardScaler; categorical imputer+OneHotEncoder(drop=first,handle_unknown=ignore). Categorical: SEX EDUCATION MARRIAGE and all PAY status columns. No hyperparameter search or calibration.
- Predeclared thresholds: 0.2, 0.3, 0.5. Select greatest validation recall subject to review rate <=30%; ties: precision, then greatest threshold. Freeze before test. If none feasible choose smallest review rate (then greatest threshold).
- Predictions data/generated/predictions_{validation,test}.csv: row_position,y_true,probability,baseline_probability (no original ID). Test predictions produced once per run after policy JSON saved.
- Exports are public aggregate CSV/JSON; images PNG; report/report.md Russian, built after computation. Report generator src/reporting.py owned by documentation agent. Lead owns remaining Python modules, README and integration (unless reassigned).
- No model tuning based on test; repeated clean runs only verify deterministic reproduction, not select model.
