import unittest
from event_extractor import finalize,speaker_gate,extract_candidates
class Tests(unittest.TestCase):
 def setUp(self):
  self.case={'id':'clip','conversation_id':'conv','interval':[0,5],'target':[1,4],'streams':{'A':[{'id':'A:0','text':'go.','start':.1,'end':2.5},{'id':'A:1','text':'No.','start':3,'end':4}],'B':[{'id':'B:0','text':'wait.','start':1,'end':3.5}]}}
 def test_bidirectional_candidates(self):
  c=extract_candidates(self.case);self.assertIn(('B','A'),[(x['initiator'],x['recipient']) for x in c]);self.assertIn(('A','B'),[(x['initiator'],x['recipient']) for x in c])
 def test_separate_reverse_and_feedback(self):
  c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='KEEP',label='BACKCHANNEL' if x['initiator']=='A' else 'INTERRUPT',merge_into='',reason='test') for x in c];r=finalize(self.case,c,{'decisions':ds},{'status':'PASS'});self.assertEqual(len(r['events']),2);self.assertEqual(len(r['filtered_candidates']),1);self.assertEqual({e['label'] for e in r['events']},{'BACKCHANNEL','INTERRUPT'})
 def test_boundary_float_tolerance(self):
  self.case['target']=[1.0000000000002,3.5];c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='KEEP',label='INTERRUPT',merge_into='',reason='test') for x in c];r=finalize(self.case,c,{'decisions':ds},{'status':'PASS'});self.assertTrue(next(e for e in r['events'] if e['initiator']=='B')['in_original_target'])
 def test_missing_candidates_rejected(self):
  with self.assertRaises(ValueError):finalize(self.case,extract_candidates(self.case),{'decisions':[]},{'status':'PASS'})
 def test_speaker_gate(self):
  self.assertEqual(speaker_gate(self.case)['status'],'UNVERIFIED');self.assertEqual(speaker_gate(self.case,{'third_speaker_present':True,'source':'verified_audio'})['status'],'EXCLUDE')
  self.case['streams']['C']=[];self.assertEqual(speaker_gate(self.case)['status'],'EXCLUDE')
 def test_unverified_not_training(self):
  c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='KEEP',label='INTERRUPT',merge_into='',reason='test') for x in c];self.assertEqual(finalize(self.case,c,{'decisions':ds},speaker_gate(self.case))['training_events'],[])
 def test_reverse_merge_rejected(self):
  c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='KEEP',label='INTERRUPT',merge_into='',reason='test') for x in c]
  parent=next(x for x in c if x['initiator']=='B')
  ds[-1].update(action='MERGE',merge_into=parent['candidate_id'])
  with self.assertRaises(ValueError):finalize(self.case,c,{'decisions':ds},{'status':'PASS'})
 def test_third_speaker_excludes_training(self):
  c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='KEEP',label='INTERRUPT',merge_into='',reason='test') for x in c];r=finalize(self.case,c,{'decisions':ds},speaker_gate(self.case,{'third_speaker_present':True}));self.assertTrue(r['events']);self.assertEqual(r['training_events'],[])
 def test_nonoverlap_continuation_is_not_lost(self):
  self.case['streams']['B'].append({'id':'B:1','text':'continuing.','start':3.6,'end':4.8})
  c=extract_candidates(self.case);self.assertTrue(any('continuing.' in x['text'] for x in c))
 def test_future_overlap_not_new_interruption(self):
  c=extract_candidates(self.case);first=next(x for x in c if x['start']==.1);self.assertTrue(first['overlap_intervals']);self.assertFalse(first['other_active_run_at_onset'])
 def test_merge_of_dropped_continuation_stays_dropped(self):
  c=extract_candidates(self.case);ds=[dict(candidate_id=x['candidate_id'],action='DROP',label='BACKCHANNEL',merge_into='',reason='test') for x in c];first=next(x for x in c if x['initiator']=='A');last=c[-1];ds[-1].update(action='MERGE',merge_into=first['candidate_id']);r=finalize(self.case,c,{'decisions':ds},{'status':'PASS'});self.assertEqual(r['events'],[])
if __name__=='__main__':unittest.main()
