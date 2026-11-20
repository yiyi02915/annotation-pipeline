import io,json,unittest
from unittest.mock import patch
from context_budget import fit_gap_payload

class BudgetTests(unittest.TestCase):
    def payload(self):
        blocks=[dict(block_id=f'B{i}',speaker='A' if i%2 else 'B',start=i,end=i+.8,
                     text='a whole sentence stays intact.',word_timing=[['sentence',i,i+.8]]) for i in range(20)]
        return dict(context=blocks,context_policy=dict(history_truncated=False),
                    target=dict(absolute_gap=[10.1,10.4]),other_near_target=blocks,other_before_right=blocks)
    def test_fit_preserves_all_history(self):
        p=self.payload()
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(dict(count=1000,max_model_len=8192)).encode())):
            value,audit=fit_gap_payload(p,'rules','http://localhost/v1/chat/completions','model')
        self.assertEqual(value,p);self.assertFalse(audit['whole_block_fallback'])
    def test_fallback_only_removes_complete_prior_blocks(self):
        p=self.payload()
        replies=[io.BytesIO(json.dumps(dict(count=n,max_model_len=8192)).encode()) for n in (9000,4000)]
        with patch('urllib.request.urlopen',side_effect=replies):
            value,audit=fit_gap_payload(p,'rules','http://localhost/v1/chat/completions','model')
        self.assertEqual(value['context'],p['context'][4:])
        self.assertEqual(len(p['context']),20)
        self.assertTrue(audit['whole_block_fallback'])
        self.assertEqual(value['target'],p['target'])
    def test_no_silent_word_truncation_if_still_too_large(self):
        replies=[io.BytesIO(json.dumps(dict(count=9000,max_model_len=8192)).encode()) for _ in range(2)]
        with patch('urllib.request.urlopen',side_effect=replies),self.assertRaisesRegex(ValueError,'no_sentence_truncation'):
            fit_gap_payload(self.payload(),'rules','http://localhost/v1/chat/completions','model')

if __name__=='__main__':unittest.main()
