import copy
import unittest
from dialogue_context import DialogueContext, apply_long_gap_policy
from gap_judge import shorter

def w(i,a,b,text): return dict(id=i,start=a,end=b,text=text)

class CompleteContextTests(unittest.TestCase):
    def setUp(self):
        self.streams={'A':[w('a',0,1,'Unfinished'),w('b',5,6,'sentence.'),w('c',12,13,'New action.')],
                      'B':[w('d',2,3,'Yes.'),w('e',8,10,'A complete answer.') ]}
        self.ctx=DialogueContext(self.streams)

    def test_no_silence_cut_and_overlap_preserved(self):
        block=self.ctx.blocks[self.ctx.by_word['a']]
        self.assertEqual([x['id'] for x in block['words']],['a','b'])
        selected=self.ctx.surrounding(5,6)
        self.assertTrue(any(b['speaker']=='B' and b['start']==2 for b in selected))

    def test_candidate_identity_unchanged(self):
        case=dict(interval=[4,7],target=[5,6],streams=copy.deepcopy(self.streams))
        result=self.ctx.bi(case)
        self.assertEqual(result['streams'],case['streams'])
        self.assertEqual(result['target'],case['target'])
        self.assertNotIn('dialogue_context',case)

    def test_full_history_and_exact_target(self):
        p=dict(focus='A',other='B',target=dict(left_word_id='a',right_word_id='b',absolute_gap=[1,5]))
        result=self.ctx.gap(p)
        self.assertEqual(result['target'],p['target'])
        self.assertEqual(result['target_right_unit']['text'],'sentence.')
        self.assertEqual(shorter(result)['context'],result['context'])
        self.assertFalse(result['context_policy']['history_truncated'])

    def test_cap_strictly_greater_and_preserves_semantics(self):
        decision=dict(status='ok',label='PAUSE',relation='CONTINUATION')
        for seconds,expected in [(3,'PAUSE'),(3.001,'TURN_TRANSITION')]:
            result=apply_long_gap_policy(decision,dict(target=dict(absolute_gap=[1,1+seconds])),3)
            self.assertEqual(result['label'],expected)
            self.assertEqual(result['semantic_label'],'PAUSE')
        self.assertEqual(decision['label'],'PAUSE')
        self.assertEqual(apply_long_gap_policy(decision,dict(target=dict(absolute_gap=[1,50])),0)['label'],'PAUSE')

    def test_error_is_not_converted_into_valid_label(self):
        self.assertEqual(apply_long_gap_policy(dict(status='error'),{},3),dict(status='error'))
        with self.assertRaises(ValueError): apply_long_gap_policy(dict(status='ok',label='PAUSE'),dict(target=dict(absolute_gap=[3,2])),3)

if __name__=='__main__': unittest.main()
