"""Fixed model, validation-only threshold policy and held-out evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import CATEGORICAL, NUMERIC, FEATURES, SEED, write_json

THRESHOLDS = [.2, .3, .5]
RULE = 'Maximize validation recall among thresholds 0.2/0.3/0.5 with review_rate <= 0.30; break ties by precision then larger threshold. If none feasible, minimize review_rate then choose larger threshold.'


def build_pipeline():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    return Pipeline([('preprocess', ColumnTransformer([('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)])),
                     ('model', LogisticRegression(C=1., max_iter=3000, solver='lbfgs', random_state=SEED))])


def metrics(y, p, threshold, model='logistic_regression'):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(model=model, threshold=float(threshold), n=len(y), positive_support=int(y.sum()),
                negative_support=int((y == 0).sum()), roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)), fpr=float(fp / (fp + tn)),
                review_rate=float(pred.mean()), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def choose_threshold(table):
    feasible = table[table.review_rate <= .30]
    if len(feasible):
        return float(feasible.sort_values(['recall', 'precision', 'threshold'], ascending=False).iloc[0].threshold)
    return float(table.sort_values(['review_rate', 'threshold'], ascending=[True, False]).iloc[0].threshold)


def calibration_table(y, p, model):
    rows = []
    edges = np.linspace(0, 1, 11)
    bins = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, 9)
    for i in range(10):
        mask = bins == i
        rows.append(dict(model=model, bin_left=edges[i], bin_right=edges[i + 1], n=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                         observed_rate=float(np.asarray(y)[mask].mean()) if mask.any() else np.nan))
    return pd.DataFrame(rows)


def export_coefficients(pipeline, root):
    pre = pipeline.named_steps['preprocess']
    names = pre.get_feature_names_out()
    coefficients = pipeline.named_steps['model'].coef_[0]
    scales = pre.named_transformers_['numeric'].named_steps['scale'].scale_
    encoder = pre.named_transformers_['categorical'].named_steps['encode']
    references = {name: int(cats[0]) for name, cats in zip(CATEGORICAL, encoder.categories_)}
    descriptions = []
    for name, scale in zip(NUMERIC, scales):
        unit = 'years' if name == 'AGE' else 'NT dollars'
        descriptions.append(f'+1 train SD = {scale:.6g} {unit}; other model covariates fixed')
    for name, cats in zip(CATEGORICAL, encoder.categories_):
        descriptions.extend([f'{name}={int(cat)} versus reference {int(cats[0])}; other model covariates fixed' for cat in cats[1:]])
    table = pd.DataFrame({'feature': names, 'coefficient': coefficients, 'odds_ratio': np.exp(coefficients), 'interpretation': descriptions})
    table.to_csv(root / 'exports/coefficients.csv', index=False, lineterminator='\n')
    write_json(root / 'exports/model_spec.json', {'features': FEATURES, 'categorical': CATEGORICAL, 'numeric': NUMERIC,
        'references': references, 'numeric_train_sd': dict(zip(NUMERIC, scales.tolist())),
        'C': 1, 'solver': 'lbfgs', 'class_weight': None, 'iterations': int(pipeline.named_steps['model'].n_iter_[0]),
        'seed': SEED, 'imputation': 'train median numeric / most frequent categorical',
        'encoding': 'one-hot drop first (smallest observed train code), unknown all-zero reference encoding with warning',
        'intercept': float(pipeline.named_steps['model'].intercept_[0])})
    return table


def evaluate(df, splits, root):
    X, y = df[FEATURES], df['default']
    assert 'ID' not in X and 'default' not in X and len(X.columns) == 23
    tr, va, te = (splits[k] for k in ['train', 'validation', 'test'])
    model = build_pipeline().fit(X.iloc[tr], y.iloc[tr])
    baseline = DummyClassifier(strategy='prior').fit(X.iloc[tr], y.iloc[tr])
    pv = model.predict_proba(X.iloc[va])[:, 1]
    bv = baseline.predict_proba(X.iloc[va])[:, 1]
    validation = pd.DataFrame([metrics(y.iloc[va], pv, t) for t in THRESHOLDS])
    validation.to_csv(root / 'exports/thresholds_validation.csv', index=False, lineterminator='\n')
    chosen = choose_threshold(validation)
    write_json(root / 'exports/policy.json', {'selected_threshold': chosen, 'rule': RULE, 'candidate_thresholds': THRESHOLDS,
        'review_capacity': .30, 'classification_rule': 'probability >= threshold', 'selection_split': 'validation',
        'baseline_classification_threshold': .5, 'test_role': 'evaluation only after policy frozen'})
    pd.DataFrame({'row_position': va, 'y_true': y.iloc[va], 'probability': pv, 'baseline_probability': bv}).to_csv(root / 'data/generated/predictions_validation.csv', index=False, lineterminator='\n')
    # Final test probabilities are computed only after the policy is written.
    pt = model.predict_proba(X.iloc[te])[:, 1]
    bt = baseline.predict_proba(X.iloc[te])[:, 1]
    pd.DataFrame({'row_position': te, 'y_true': y.iloc[te], 'probability': pt, 'baseline_probability': bt}).to_csv(root / 'data/generated/predictions_test.csv', index=False, lineterminator='\n')
    test_thresholds = pd.DataFrame([metrics(y.iloc[te], pt, t) for t in THRESHOLDS])
    test_thresholds.to_csv(root / 'exports/thresholds_test.csv', index=False, lineterminator='\n')
    result = pd.DataFrame([metrics(y.iloc[te], bt, .5, 'baseline_prior'), metrics(y.iloc[te], pt, chosen)])
    result.to_csv(root / 'exports/metrics_test.csv', index=False, lineterminator='\n')
    calibration = pd.concat([calibration_table(y.iloc[te], pt, 'logistic_regression'), calibration_table(y.iloc[te], bt, 'baseline_prior')], ignore_index=True)
    calibration.to_csv(root / 'exports/calibration_test.csv', index=False, lineterminator='\n')
    groups = []
    test_df = df.iloc[te].copy()
    test_df['age_band'] = pd.cut(test_df.AGE, [0, 30, 40, 50, 60, np.inf], labels=['<=30', '31-40', '41-50', '51-60', '>60'])
    for group in ['SEX', 'age_band', 'MARRIAGE']:
        for category in sorted(test_df[group].dropna().unique()):
            mask = (test_df[group] == category).to_numpy()
            # Tiny groups may have only one target class: compute classification counts directly.
            yy, pp = y.iloc[te].to_numpy()[mask], pt[mask]
            tn, fp, fn, tp = confusion_matrix(yy, pp >= chosen, labels=[0, 1]).ravel()
            groups.append(dict(group=group, category=str(category), n=len(yy), positives=int(yy.sum()),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), precision=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan, fpr=fp/(fp+tn) if fp+tn else np.nan,
                review_rate=float((pp >= chosen).mean()), brier=float(brier_score_loss(yy, pp)), rare_group=len(yy)<100))
    pd.DataFrame(groups).to_csv(root / 'exports/group_errors_test.csv', index=False, lineterminator='\n')
    coefficients = export_coefficients(model, root)
    return {'model': model, 'y_test': y.iloc[te].to_numpy(), 'probabilities': pt, 'baseline': bt,
            'threshold': chosen, 'coefficients': coefficients, 'metrics': result, 'calibration': calibration}
