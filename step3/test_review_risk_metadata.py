import unittest
from risk_annotations import tag_pipeline_result


class ReviewRiskMetadata(unittest.TestCase):
    def sample(self):
        return dict(events=[dict(event_id='event', candidate_ids=['a'], label='BACKCHANNEL', start=1, end=2)],
                    gaps=[], windows=[dict(target=[0, 3], candidate_intervals={'a':[1,2], 'context':[4,5]},
                      audio_votes=[dict(candidate_id='a', mix=dict(answer='BACKCHANNEL', probs={'BACKCHANNEL':.9,'INTERRUPT':.1}))],
                      decisions=[], context_reviews=[
                        dict(candidate_id='a', changed=False, status='ok', after=dict(action='KEEP',label='BACKCHANNEL')),
                        dict(candidate_id='context',changed=True,status='ok',after=dict(action='DROP',label='BACKCHANNEL'))],
                      response_audits=[dict(candidate_id='a',removal_blocked_reason='active_recipient_clause_preserves_feedback')])])

    def test_blocked_change_is_risk_not_label_override_or_double_count(self):
        result=tag_pipeline_result(self.sample())
        self.assertEqual(result['events'][0]['label'],'BACKCHANNEL')
        self.assertIn('semantic_review_timing_conflict',result['events'][0]['quality']['risk_reasons'])
        self.assertEqual(result['risk_summary']['flagged_annotations'],1)
        self.assertEqual(result['risk_summary']['review_audit_items'],1)
        self.assertFalse(any(r['id']=='context' for r in result['risk_items']))

    def test_changed_and_failed_flags_both_survive(self):
        result=self.sample();result['windows'][0]['context_reviews'][0].update(changed=True,status='error_retained_original')
        result=tag_pipeline_result(result)
        audit=next(r for r in result['risk_items'] if r['kind']=='context_review')
        self.assertIn('label_changed_on_context_review',audit['risk_reasons'])
        self.assertIn('context_review_failure',audit['risk_reasons'])


if __name__=='__main__':unittest.main()
