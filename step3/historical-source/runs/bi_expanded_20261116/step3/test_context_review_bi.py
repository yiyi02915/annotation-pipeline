import unittest
from context_review_bi import eligible,apply_review,recipient_key,feedback_parent

class GroundedReview(unittest.TestCase):
    def setUp(self):
        self.c=dict(other_active_run_at_onset=True,overlap_seconds=1.6,
            recipient_is_question=True,recipient_tail_seconds=.1,text='Yes, I worked there.',
            recipient_clause_at_onset=dict(text='You worked there?'))
        self.d=dict(action='KEEP',label='INTERRUPT',merge_into='',reason='initial')
    def test_question_tail_eligible_despite_later_overlap(self):
        self.assertEqual(eligible(self.c,self.d),['NORMAL_ANSWER'])
    def test_earlier_question_entry_cannot_be_suppressed(self):
        self.c['recipient_tail_seconds']=.8
        self.assertEqual(eligible(self.c,self.d),[])
    def test_non_question_substantive_entry_is_protected(self):
        self.c['recipient_is_question']=False
        self.assertEqual(eligible(self.c,self.d),[])
    def test_normal_answer_needs_grounded_quotes(self):
        answer=dict(finding='NORMAL_ANSWER',target_quote='invented',recipient_quote='You worked there?',reason='answer')
        with self.assertRaisesRegex(ValueError,'not_grounded'):apply_review(self.c,self.d,answer,['NORMAL_ANSWER'])
        self.assertEqual(self.d['action'],'KEEP')
    def test_finding_cannot_override_allowed_change(self):
        answer=dict(finding='NEW_FEEDBACK',target_quote='Yes',recipient_quote='You',reason='receipt')
        with self.assertRaisesRegex(ValueError,'not_allowed'):apply_review(self.c,self.d,answer,['NORMAL_ANSWER'])
    def test_grounded_answer_drops_only_exact_candidate(self):
        answer=dict(finding='NORMAL_ANSWER',target_quote='Yes',recipient_quote='You worked there?',reason='answer in final slot')
        self.assertTrue(apply_review(self.c,self.d,answer,['NORMAL_ANSWER']))
        self.assertEqual(self.d['action'],'DROP')
    def test_new_receipt_is_kept_as_backchannel(self):
        self.c['text']='All right.';self.d['action']='DROP'
        answer=dict(finding='NEW_FEEDBACK',target_quote='All right.',recipient_quote='You worked there?',reason='receipt')
        apply_review(self.c,self.d,answer,['NEW_FEEDBACK'])
        self.assertEqual((self.d['action'],self.d['label']),('KEEP','BACKCHANNEL'))
    def test_no_overlap_does_not_trigger_review(self):
        self.c['overlap_seconds']=0
        self.assertEqual(eligible(self.c,self.d),[])
    def test_recovery_cannot_undo_existing_terminal_answer_guard(self):
        self.c.update(text='Yes.',terminal_question_answer=True)
        self.d.update(action='DROP',label='BACKCHANNEL')
        self.assertEqual(eligible(self.c,self.d),[])
    def test_own_fragments_are_not_feedback_recovery_targets(self):
        self.d['action']='DROP'
        for text in ['and','I','Do','It comes and goes.']:
            self.c['text']=text
            self.assertEqual(eligible(self.c,self.d),[])
    def test_same_receipt_source_requires_same_direction_and_time(self):
        a=dict(initiator='A',recipient='B',recipient_clause_at_onset=dict(text='Yeah.',start=1,end=2))
        b=dict(a,recipient_clause_at_onset=dict(text='Yeah.',start=3,end=4))
        self.assertNotEqual(recipient_key(a),recipient_key(b))
    def test_feedback_continuity_requires_nearby_same_recipient_contribution(self):
        c=dict(initiator='A',recipient='B',start=2.2,recipient_clause_at_onset=dict(text='long',start=1,end=5))
        active={recipient_key(c):('earlier',2)}
        self.assertEqual(feedback_parent(c,active),'earlier')
        c['start']=3
        self.assertEqual(feedback_parent(c,active),'')

if __name__=='__main__':unittest.main()
