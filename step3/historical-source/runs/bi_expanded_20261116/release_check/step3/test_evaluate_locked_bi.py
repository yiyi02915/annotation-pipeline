import unittest
from evaluate_locked_bi import evaluate


class LockedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.ref = {'cases': {'sample': {'events': [
            {'candidate_id': 'root', 'label': 'INTERRUPT'},
            {'candidate_id': 'child', 'label': 'NEITHER', 'function': 'continuation'}],
            'structure_checks': [{'parent': 'root', 'child': 'child',
                                  'label': 'INTERRUPT', 'kind': 'same_interruption'}]}}}

    def test_merge_and_deletion_have_same_onset_label_but_different_link_quality(self):
        before = {'sample': {'status': 'ok', 'events': [
            {'label': 'INTERRUPT', 'candidate_ids': ['root']}]}}
        after = {'sample': {'status': 'ok', 'events': [
            {'label': 'INTERRUPT', 'candidate_ids': ['root', 'child']}]}}
        out = evaluate(self.ref, before, after)
        self.assertEqual(out['metrics']['before']['all']['correct'], 2)
        self.assertEqual(out['metrics']['after']['all']['correct'], 2)
        self.assertEqual(out['metrics']['before']['structures']['same_interruption']['correct'], 0)
        self.assertEqual(out['metrics']['after']['structures']['same_interruption']['correct'], 1)

    def test_failed_prediction_does_not_become_correct_neither(self):
        failed = {'sample': {'status': 'error'}}
        out = evaluate(self.ref, failed, failed)
        self.assertEqual(out['metrics']['after']['all']['correct'], 0)
        self.assertEqual(out['metrics']['after']['all']['failed'], 2)

    def test_missing_windows_rejected_instead_of_inflating_agreement(self):
        with self.assertRaises(ValueError):
            evaluate(self.ref, {}, {})


if __name__ == '__main__':
    unittest.main()
