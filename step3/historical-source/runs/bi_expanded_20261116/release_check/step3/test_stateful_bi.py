import unittest
from stateful_bi import add_onset_features,to_actions

def candidate(uid,start,speaker='A',recipient='B'):
    return dict(candidate_id=uid,start=start,end=start+.5,initiator=speaker,recipient=recipient,
                speech_run_start=0,overlap_intervals=[[start,start+.1],[start+2,start+3]])
def state(uid,function,entry='NEW',parent='',prior='OPEN'):
    return dict(candidate_id=uid,function=function,entry=entry,parent=parent,prior=prior,reason='evidence')

class TypedEvents(unittest.TestCase):
    def test_feedback_maps_to_keep_not_drop(self):
        self.assertEqual(to_actions([candidate('a',1)],[state('a','FEEDBACK')])['decisions'][0]['action'],'KEEP')
    def test_normal_answer_is_not_an_interruption(self):
        d=to_actions([candidate('a',1)],[state('a','ORDINARY_RESPONSE',prior='FINAL_ANSWER_SLOT')])['decisions'][0]
        self.assertEqual(d['action'],'DROP')
    def test_same_event_continuation_preserves_parent(self):
        d=to_actions([candidate('a',1),candidate('b',2)],
                     [state('a','FLOOR_CLAIM'),state('b','FLOOR_CLAIM','CONTINUATION','a')])['decisions']
        self.assertEqual([x['action'] for x in d],['KEEP','MERGE'])
    def test_reverse_direction_cannot_merge(self):
        with self.assertRaisesRegex(ValueError,'invalid_semantic_continuation_parent'):
            to_actions([candidate('a',1),candidate('b',2,'B','A')],
                       [state('a','FLOOR_CLAIM'),state('b','FLOOR_CLAIM','CONTINUATION','a')])
    def test_existing_interruption_can_continue_after_recipient_yields(self):
        d=to_actions([candidate('a',1),candidate('b',2)],
                     [state('a','FLOOR_CLAIM'),state('b','FLOOR_CLAIM','CONTINUATION','a',prior='ENDED')])['decisions']
        self.assertEqual(d[1]['action'],'MERGE')
    def test_real_reverse_entry_and_nested_feedback_stay_separate(self):
        candidates=[candidate('a',1),candidate('b',2,'B','A'),candidate('c',3)]
        d=to_actions(candidates,[state('a','FLOOR_CLAIM'),state('b','FLOOR_CLAIM'),state('c','FEEDBACK')])['decisions']
        self.assertEqual([x['action'] for x in d],['KEEP']*3)
    def test_dropped_parent_cannot_support_event_continuation(self):
        with self.assertRaises(ValueError):
            to_actions([candidate('a',1),candidate('b',2)],[state('a','OWN_CONTINUATION'),state('b','FLOOR_CLAIM','CONTINUATION','a')])
    def test_claim_after_yield_is_structural_error_not_silent_relabel(self):
        with self.assertRaisesRegex(ValueError,'floor_claim_requires_open'):
            to_actions([candidate('a',1)],[state('a','FLOOR_CLAIM',prior='ENDED')])
    def test_later_overlap_does_not_inflate_initial_overlap(self):
        c=candidate('a',1);c['recipient_clause_at_onset']=dict(end=1.2,text='A question?')
        add_onset_features([c])
        self.assertEqual(c['overlap_with_onset_clause'],.1)
        self.assertEqual(c['recipient_tail_seconds'],.2)
        self.assertTrue(c['recipient_is_question'])
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            to_actions([candidate('a',1),candidate('b',2)],[state('a','FEEDBACK'),state('a','FEEDBACK')])

if __name__=='__main__':unittest.main()
