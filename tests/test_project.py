"""Meaningful contract checks: split isolation, train-only preprocessing, metric identities."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.data import FEATURES, NUMERIC, make_splits
from src.modeling import build_pipeline, choose_threshold, metrics, calibration_table

ROOT = Path(__file__).resolve().parents[1]


class MethodologyTests(unittest.TestCase):
    def test_threshold_capacity_and_tie_rule(self):
        table = pd.DataFrame({'threshold': [.2, .3, .5], 'recall': [.9, .7, .4], 'precision': [.4, .6, .8], 'review_rate': [.5, .28, .1]})
        self.assertEqual(choose_threshold(table), .3)
        table.loc[2, ['recall', 'precision']] = [.7, .6]
        self.assertEqual(choose_threshold(table), .5)

    def test_known_confusion_counts_and_denominators(self):
        result = metrics([0, 0, 0, 1, 1], [.1, .8, .2, .7, .3], .5)
        self.assertEqual([result[k] for k in ['tn', 'fp', 'fn', 'tp']], [2, 1, 1, 1])
        self.assertAlmostEqual(result['recall'], .5)
        self.assertAlmostEqual(result['fpr'], 1/3)
        self.assertAlmostEqual(result['review_rate'], 2/5)

    def test_train_only_scaler_unknown_category(self):
        X = pd.DataFrame({c: [1., 2., 3., 4., 5., 6.] for c in FEATURES})
        pipe = build_pipeline().fit(X, [0, 1, 0, 1, 0, 1])
        scaler = pipe.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        before = scaler.mean_.copy()
        heldout = X.iloc[:2].copy()
        heldout.loc[:, 'LIMIT_BAL'] = 1000000.
        heldout.loc[:, 'SEX'] = 999.
        with self.assertWarns(UserWarning):
            probabilities = pipe.predict_proba(heldout)
        np.testing.assert_array_equal(scaler.mean_, before)
        np.testing.assert_allclose(before, np.repeat(3.5, len(NUMERIC)))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_calibration_includes_boundary_probabilities(self):
        table = calibration_table(np.array([0, 1, 1]), np.array([0., .1, 1.]), 'test')
        self.assertEqual(table.n.sum(), 3)
        self.assertEqual(table.iloc[-1].n, 1)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'data/generated/predictions_test.csv').exists():
            raise unittest.SkipTest('Run python -m src.run first to validate generated artifacts')

    def test_split_isolation_and_sizes(self):
        membership = pd.read_csv(ROOT / 'data/generated/splits.csv')
        self.assertEqual(len(membership), 30000)
        self.assertTrue(membership.row_position.is_unique)
        self.assertEqual(membership.groupby('split').size().to_dict(), {'test': 6000, 'train': 18000, 'validation': 6000})
        for name in ['validation', 'test']:
            pred = pd.read_csv(ROOT / f'data/generated/predictions_{name}.csv')
            self.assertNotIn('ID', pred.columns)
            self.assertEqual(set(pred.row_position), set(membership.query('split == @name').row_position))
            self.assertTrue(pred.probability.between(0, 1).all())

    def test_exported_metrics_recomputed(self):
        p = pd.read_csv(ROOT / 'data/generated/predictions_test.csv')
        saved = pd.read_csv(ROOT / 'exports/metrics_test.csv')
        for _, row in saved.iterrows():
            column = 'baseline_probability' if row.model == 'baseline_prior' else 'probability'
            expected = metrics(p.y_true, p[column], row.threshold, row.model)
            for key, value in expected.items():
                if key != 'model':
                    self.assertAlmostEqual(row[key], value, places=10, msg=key)

    def test_selected_policy_comes_from_validation(self):
        policy = json.loads((ROOT / 'exports/policy.json').read_text(encoding='utf-8'))
        val = pd.read_csv(ROOT / 'exports/thresholds_validation.csv')
        self.assertEqual(policy['selected_threshold'], choose_threshold(val))
        self.assertEqual(policy['selection_split'], 'validation')

    def test_group_counts_and_calibration_partition(self):
        groups = pd.read_csv(ROOT / 'exports/group_errors_test.csv')
        for _, group in groups.groupby('group'):
            self.assertEqual(group.n.sum(), 6000)
            np.testing.assert_array_equal(group.n, group.tp + group.fp + group.tn + group.fn)
        bins = pd.read_csv(ROOT / 'exports/calibration_test.csv')
        self.assertTrue((bins.groupby('model').n.sum() == 6000).all())


if __name__ == '__main__':
    unittest.main()
